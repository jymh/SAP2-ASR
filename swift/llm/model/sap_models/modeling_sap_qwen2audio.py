import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import torch.utils.checkpoint
import torch
from torch import nn


from transformers import Qwen2AudioForConditionalGeneration, Qwen2AudioConfig
from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioCausalLMOutputWithPast
from transformers.modeling_outputs import ModelOutput
from transformers.cache_utils import Cache



def fix_window_size_pooling(hidden_states, attention_mask, weights):
    bsz, pooled_length, window_size, hidden_size = hidden_states.size()
    scatter_matrix = torch.zeros_like(attention_mask)
    scatter_matrix[..., ::window_size] = 1
    scatter_index = scatter_matrix.cumsum(dim=-1) - 1

    hidden_states_after_weighting = (hidden_states * weights).view(bsz, -1, hidden_size)
    pooling_hidden_states = torch.zeros([bsz, pooled_length, hidden_size], device=hidden_states.device).to(hidden_states.dtype)
    pooling_hidden_states.scatter_add_(1, scatter_index[..., None].repeat(1, 1, hidden_size), hidden_states_after_weighting)

    pooling_attention_mask = torch.zeros([bsz, pooled_length], device=hidden_states.device).to(attention_mask.dtype)
    pooling_attention_mask.scatter_add_(1, scatter_index, attention_mask)
    pooling_attention_mask = pooling_attention_mask.greater(0).to(attention_mask.dtype)
    
    return pooling_hidden_states, pooling_attention_mask

class Qwen2AudioSAP2PoolingLayer(nn.Module):
    def __init__(self, compressor_hidden_size, num_attention_heads, **kwargs):
        super().__init__()
        self.hidden_size = compressor_hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        # self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        # self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        # self.semantic_alignment_layer = nn.Linear(self.hidden_size, 4096)  # project to LLM dimension
        # self.contxt_layernorm = nn.LayerNorm(compressor_hidden_size)
        # self.audio_layernorm = nn.LayerNorm(compressor_hidden_size)
        
        
    def forward(self, audio_hidden_states, contxt_hidden_states, enc_audio_mask, enc_contxt_mask, window_size=None, **kwargs):
        if window_size is None:
            if self.args.random_pool_window_size:
                window_size = random.choice(self.args.cand_pool_window_sizes)
            else:
                window_size = self.args.pool_window_size
                
        bsz, contxt_len, hidden_size = contxt_hidden_states.size()
        if contxt_len % window_size != 0:
            def padding(tensor, shape):
                return torch.cat([tensor, torch.zeros(shape, dtype=tensor.dtype, device=tensor.device,)], dim=1)
            
            padding_length = window_size - contxt_len % window_size
            contxt_hidden_states = padding(contxt_hidden_states, shape=(bsz, padding_length, hidden_size))
            enc_contxt_mask = padding(enc_contxt_mask, shape=(bsz, padding_length))
            contxt_len = enc_contxt_mask.size(1)
        
        

        audio_hidden_states = audio_hidden_states.masked_fill(~enc_audio_mask[..., None].bool(), 0.0)
        
        query_states = audio_hidden_states.contiguous().reshape(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2) # query_states: (bsz, num_heads, audio_length, head_dim)
        key_states = contxt_hidden_states.contiguous().reshape(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2) # key_states: (bsz, num_heads, contxt_length, head_dim)
        # value_states = contxt_hidden_states.reshape(bsz, -1, self.num_heads, self.head_dim)
        
        pooling_weights = torch.einsum('bnqh,bnkh->bnqk', query_states, key_states) / math.sqrt(self.head_dim) # pooling_weights: (bsz, num_heads, audio_length, contxt_length)
        pooling_weights.masked_fill(~enc_audio_mask[:, None, :, None].bool(), torch.finfo(query_states.dtype).min) # pooling_weights: (bsz, num_heads, audio_length, contxt_length)
        
        pooling_weights = pooling_weights.softmax(dim=-2)
        pooling_weights = pooling_weights.sum(dim=-2) / enc_audio_mask[..., None, None].sum(dim=1) # (bsz, num_heads, contxt_len)
        pooling_weights.masked_fill(~enc_contxt_mask.unsqueeze(1).bool(), torch.finfo(query_states.dtype).min)
        pooling_weights = pooling_weights.reshape(bsz, self.num_heads, -1, window_size)
        pooling_weights = pooling_weights.softmax(dim=-1) # (bsz, num_heads, pooled_length, window_size)
        
        combined_pooling_weights = pooling_weights.permute(0, 2, 3, 1) # (bsz, pooled_length, window_size, num_heads)
        combined_pooling_weights = combined_pooling_weights[..., None].repeat(1, 1, 1, 1, self.head_dim).contiguous().reshape(bsz, -1, window_size, self.hidden_size)
        combined_value_states = contxt_hidden_states.reshape(bsz, -1, window_size, self.hidden_size)
        
        pooling_hidden_states = (combined_value_states * combined_pooling_weights).sum(dim=2)
        pooling_attention_mask = enc_contxt_mask.reshape(bsz, -1, window_size).sum(dim=2).greater(0).to(enc_contxt_mask.dtype)
        
        # pooling_hidden_states = (combined_pooling_weights * combined_pooling_weights).reshape(bsz, -1, hidden_size)
        # pooling_attention_mask = enc_contxt_mask
        
        # pooling_hidden_states = self.semantic_alignment_layer(pooling_hidden_states)

        return pooling_hidden_states, pooling_attention_mask
        



class SAP2Qwen2AudioForConditionalGeneration(Qwen2AudioForConditionalGeneration):
    def __init__(self, config, sap_window_size=None, **kwargs):
        super().__init__(config)
        
        self.sap2_pooling_layer = Qwen2AudioSAP2PoolingLayer(**kwargs)
        self.sap_window_size = sap_window_size
        
        self.audio_bos_token_id = 151647  # <|audio_bos|>
        self.audio_eos_token_id = 151648  # <|audio_eos|>
        self.context_bos_token_id = None  # <|startofcontext|>
        self.context_eos_token_id = None  # <|endofcontext|>
        
        self.text_pad_token_id = self.config.text_config.bos_token_id # <|endoftext|>
        
        self.post_init()
        
    def _merge_input_ids_with_audio_features(
        self, audio_features, num_audio_tokens, inputs_embeds, input_ids, attention_mask, labels
    ):
        """
        Merge input_ids with with audio features into final embeddings

        Args:
            audio_features (`torch.Tensor` of shape `(num_audios, max_audio_tokens, embed_dim)`):
                All audio vectors of all audios in the batch
            num_audio_tokens (`torch.LongTensor` of shape `(num_audios)`):
                The length of audio embeddings of each audio as stacked in `audio_features`
            inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, embed_dim)`):
                Token embeddings before merging with audio embeddings
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input_ids of tokens, possibly filled with audio token
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Mask to avoid performing attention on padding token indices.
            labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*)
                labels need to be recalculated to support training (if provided)
        Returns:
            final_embedding, final_attention_mask, final_labels, position_ids, final_input_ids

        Explanation:
            each audio has variable length embeddings, with length specified by num_audio_tokens
            audio_features is concatenation of all audio embed vectors
            task: fill each <|AUDIO|> with the correct number of audio embeddings
            Example:
                X (5 tokens), Y (3 tokens), Z (8 tokens)
                X, Y are in the same sequence (in-context learning)
            if right padding
                input_ids: [
                    a b c d e f X g h i j k Y l m
                    o p q r Z s t u v _ _ _ _ _ _
                ]
                input_ids should be: [
                    a b c d e f X X X X X g h i j k Y Y Y l m
                    o p q r Z Z Z Z Z Z Z Z s t u v _ _ _ _ _
                ]
                labels should be: [
                    a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                    o p q r _ _ _ _ _ _ _ _ s t u v _ _ _ _ _
                ]
            elif left padding
                input_ids: [
                    a b c d e f X g h i j k Y l m
                    _ _ _ _ _ _ o p q r Z s t u v
                ]
                input_ids should be: [
                    a b c d e f X X X X X g h i j k Y Y Y l m
                    _ _ _ _ _ o p q r Z Z Z Z Z Z Z Z s t u v
                ]
                labels should be: [
                    a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                    _ _ _ _ _ o p q r _ _ _ _ _ _ _ _ s t u v
                ]
            Edge cases:
                * If tokens are same but audio token sizes are different, then cannot infer left or right padding
                ```python
                url1 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"
                audio1, _ = librosa.load(BytesIO(urlopen(url1).read()), sr=processor.feature_extractor.sampling_rate)
                url2 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/f2641_0_throatclearing.wav"
                audio2, _ = librosa.load(BytesIO(urlopen(url2).read()), sr=processor.feature_extractor.sampling_rate)
                prompts = [
                    "[INST] <|AUDIO|>\nWhat is that in this audio? [/INST]",
                    "[INST] <|AUDIO|>\nWhat is that in this audio? [/INST]",
                ]
                inputs = processor(text=prompts, audios=[audio1, audio2], return_tensors='pt', padding=True).to("cuda")
                    audio1 has 101 tokens, while audio2 has 72 tokens
                ```

                input_ids: [
                    a b c d X g h
                    i j Y k l m n
                ]
                where X is 3 tokens while Y is 5, this mean after merge
                if left-padding (batched generation)
                    input_ids should be: [
                        _ _ a b c d X X X g h
                        i j Y Y Y Y Y k l m n
                    ]
                elif (right padding) (training)
                    input_ids should be: [
                        a b c d X X X g h _ _
                        i j Y Y Y Y Y k l m n
                    ]
        """
        num_audios, max_audio_tokens, embed_dim = audio_features.shape
        audio_features_mask = torch.arange(max_audio_tokens).expand(num_audios, max_audio_tokens).to(
            num_audio_tokens.device
        ) < num_audio_tokens.unsqueeze(1)
        masked_audio_features = audio_features[audio_features_mask].reshape(-1, embed_dim)
        
        batch_size, sequence_length = input_ids.shape
        _left_padding = torch.any(attention_mask[:, 0] == 0)
        _right_padding = torch.any(attention_mask[:, -1] == 0)

        left_padding = True
        if batch_size > 1:
            if _left_padding and not _right_padding:
                left_padding = True
            elif not _left_padding and _right_padding:
                left_padding = False
            elif not _left_padding and not _right_padding:
                # both side is 1, so cannot tell
                left_padding = self.padding_side == "left"
            else:
                # invalid attention_mask
                raise ValueError(f"both side of attention_mask has zero, invalid. {attention_mask}")

        # 1. Create a mask to know where special audio tokens are
        special_audio_token_mask = input_ids == self.config.audio_token_index
        num_special_audio_tokens = torch.sum(special_audio_token_mask, dim=-1)

        # In case the Audio model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        attention_mask = attention_mask.to(target_device)
        input_ids = input_ids.to(target_device)
        num_audio_tokens = num_audio_tokens.to(target_device)
        # batch_indices, non_audio_indices = torch.where(
        #     (input_ids != self.config.audio_token_index) & ((attention_mask == 1) | (attention_mask == -1))
        # )
        batch_indices, non_audio_indices = torch.where(
            (input_ids != self.config.audio_token_index) & (attention_mask == 1)
        )

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged audio-text sequence.
        # `special_audio_token_mask` identifies audio tokens. Each audio token will be replaced by `audio_feat_lengths - 1` text tokens.
        # `torch.cumsum` computes how each audio token shifts subsequent text token positions.
        token_placeholder_num = torch.zeros_like(input_ids)
        token_placeholder_num[special_audio_token_mask] = num_audio_tokens.long() - 1
        token_placeholder_num = token_placeholder_num + 1
        new_token_positions = torch.cumsum(token_placeholder_num, -1) - 1
        max_token_num = token_placeholder_num.sum(-1).max()
        nb_audio_pad = max_token_num - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_audio_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_audio_indices]
        batch_indices, non_audio_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_audio_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_token_num, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_token_num, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        final_input_ids = torch.full(
            (batch_size, max_token_num), self.pad_token_id, dtype=input_ids.dtype, device=inputs_embeds.device
        )

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<audio>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the audio features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_audio_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_audio_indices]
        # final_attention_mask = torch.where(final_attention_mask == -1, torch.tensor(0, dtype=attention_mask.dtype), final_attention_mask).to(final_attention_mask.device)
        final_input_ids[batch_indices, text_to_overwrite] = input_ids[batch_indices, non_audio_indices]
        final_labels = None
        if labels is not None:
            labels = labels.to(target_device)
            final_labels = torch.full_like(final_attention_mask, self.config.ignore_index).to(torch.long)
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_audio_indices]

        # 5. Fill the embeddings corresponding to the audios. Anything that is still zeros needs filling
        audio_to_overwrite = torch.full(
            (batch_size, max_token_num), True, dtype=torch.bool, device=inputs_embeds.device
        )
        audio_to_overwrite[batch_indices, text_to_overwrite] = False
        seq_indices = torch.arange(max_token_num).unsqueeze(0).to(target_device)
        seq_indices = seq_indices.expand(batch_size, max_token_num)

        if left_padding:
            # exclude padding on the left
            max_token_num = max_token_num.to(target_device)
            val = (max_token_num - seq_indices) <= (
                token_placeholder_num.sum(-1) - (attention_mask == 0).long().sum(-1)
            )[:, None]
        else:
            # exclude padding on the right
            val = seq_indices < (token_placeholder_num.sum(-1) - (attention_mask == 0).long().sum(-1))[:, None]

        audio_to_overwrite &= val

        if audio_to_overwrite.sum() != num_audio_tokens.sum():
            raise ValueError(
                f"The input provided to the model are wrong. The number of audio tokens is {num_special_audio_tokens} while"
                f" the number of audio given to the model is {num_audios}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[audio_to_overwrite] = (
            masked_audio_features.contiguous().reshape(-1, embed_dim).to(target_device)
        )
        final_attention_mask |= audio_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)
        
        

        return final_embedding, final_attention_mask, final_labels, position_ids, final_input_ids

    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_features: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Qwen2AudioCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from io import BytesIO
        >>> from urllib.request import urlopen
        >>> import librosa
        >>> from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        >>> model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B")

        >>> prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Generate the caption in English:"
        >>> url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"
        >>> audio, _ = librosa.load(BytesIO(urlopen(url).read()), sr=self.processor.feature_extractor.sampling_rate)

        >>> inputs = processor(text=prompt, audios=audio, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=30)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Generate the caption in English: Glass is breaking."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        target_device = self.audio_tower.device

        if input_features is not None:
            input_features = input_features.to(target_device)
            feature_attention_mask = feature_attention_mask.to(target_device)

        
        if inputs_embeds is None:
            # # 1. Extract the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)
            

            # 2. Merge text and audios
            if input_features is not None and input_ids.shape[1] != 1:
                
                # extract context and apply speech-guided embedding
                batch_size, sequence_length = input_ids.shape
            
                '''
                training:
                input_ids: <|im_start|>system\n system prompt <|im_end|>\n<|im_start|>user\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n PROMPT CONTEXT<|im_end|>\n<|im_start|>assistant\n VALID LABEL...
                inference:
                input_ids: <|im_start|>system\n system prompt <|im_end|>\n<|im_start|>user\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n PROMPT CONTEXT<|im_end|>\n<|im_start|>assistant\n
                '''
                
                prompt_start_positions = torch.where(input_ids == self.context_bos_token_id)[1].long().to(input_ids.device) + 1
                prompt_eos_positions = torch.where(input_ids == self.context_eos_token_id)[1].long().to(input_ids.device)

                # Prepare input for sap compression, only compress context prompts.
                input_ids_before_prompt = [input_ids[i, :prompt_start_positions[i]] for i in range(batch_size)]
                input_ids_prompt = [input_ids[i, prompt_start_positions[i]:prompt_eos_positions[i]] for i in range(batch_size)]
                input_ids_after_prompt = [input_ids[i, prompt_eos_positions[i]:] for i in range(batch_size)]
                
                inputs_embeds_before_prompt = [inputs_embeds[i, :prompt_start_positions[i]] for i in range(batch_size)]
                inputs_embeds_prompt = [inputs_embeds[i, prompt_start_positions[i]:prompt_eos_positions[i]] for i in range(batch_size)]
                inputs_embeds_after_prompt = [inputs_embeds[i, prompt_eos_positions[i]:] for i in range(batch_size)]
                
                attention_mask_before_prompt = [attention_mask[i, :prompt_start_positions[i]] for i in range(batch_size)]
                attention_mask_prompt = [attention_mask[i, prompt_start_positions[i]:prompt_eos_positions[i]] for i in range(batch_size)]
                attention_mask_after_prompt = [attention_mask[i, prompt_eos_positions[i]:] for i in range(batch_size)]
                
                if labels is not None:
                    labels_before_prompt = [labels[i, :prompt_start_positions[i]] for i in range(batch_size)]
                    labels_prompt = [labels[i, prompt_start_positions[i]:prompt_eos_positions[i]] for i in range(batch_size)]
                    labels_after_prompt = [labels[i, prompt_eos_positions[i]:] for i in range(batch_size)]
                
                max_prmopt_length = max([seq.size(0) for seq in input_ids_prompt])
                padded_inputs_embeds_prompt = torch.full((len(input_ids_prompt), max_prmopt_length, inputs_embeds.size(2)), fill_value=0, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                padded_input_ids_prompt = torch.full((len(input_ids_prompt), max_prmopt_length), fill_value=0, dtype=input_ids.dtype, device=input_ids.device)
                for i, seq in enumerate(inputs_embeds_prompt):
                    padded_inputs_embeds_prompt[i, :seq.size(0)] = seq
                for i, seq in enumerate(input_ids_prompt):
                    padded_input_ids_prompt[i, :seq.size(0)] = seq
                prompt_attention_mask = (padded_input_ids_prompt != 0).long()
                    
                
                audio_feat_lengths, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(
                    feature_attention_mask.sum(-1)
                )
                batch_size, _, max_mel_seq_len = input_features.shape
                max_seq_len = (max_mel_seq_len - 2) // 2 + 1
                # Create a sequence tensor of shape (batch_size, max_seq_len)
                seq_range = (
                    torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype, device=audio_feat_lengths.device)
                    .unsqueeze(0)
                    .expand(batch_size, max_seq_len)
                )
                lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
                # Create mask
                padding_mask = seq_range >= lengths_expand

                audio_attention_mask_ = padding_mask.reshape(batch_size, 1, 1, max_seq_len).expand(
                    batch_size, 1, max_seq_len, max_seq_len
                )
                audio_attention_mask = audio_attention_mask_.to(
                    dtype=self.audio_tower.conv1.weight.dtype, device=self.audio_tower.conv1.weight.device
                )
                audio_attention_mask[audio_attention_mask_] = float("-inf")

                audio_outputs = self.audio_tower(input_features, attention_mask=audio_attention_mask)
                selected_audio_feature = audio_outputs.last_hidden_state
                audio_features = self.multi_modal_projector(selected_audio_feature)
                
                num_audios, max_audio_tokens, embed_dim = audio_features.shape
                audio_features_mask = torch.arange(max_audio_tokens).expand(num_audios, max_audio_tokens).to(
                    audio_output_lengths.device
                ) < audio_output_lengths.unsqueeze(1)
                
                
                # sap compress
                sap_inputs_embeds, sap_attention_mask = self.sap2_pooling_layer(audio_features, padded_inputs_embeds_prompt, audio_features_mask, prompt_attention_mask, window_size=self.sap_window_size)
                sap_inputs_embeds, sap_attention_mask = sap_inputs_embeds.to(inputs_embeds.device), sap_attention_mask.to(inputs_embeds.device)
                sap_prompt_lengths = sap_attention_mask.sum(dim=-1).to(inputs_embeds.device)     
                     
                before_prompt_lengths = torch.tensor([seq.size(0) for seq in input_ids_before_prompt], device=inputs_embeds.device)
                after_prompt_lengths = torch.tensor([seq.size(0) for seq in input_ids_after_prompt], device=inputs_embeds.device)

                final_lengths = before_prompt_lengths + after_prompt_lengths + sap_prompt_lengths
                max_final_length = final_lengths.max().item()
                
                final_input_embeds = torch.zeros(batch_size, max_final_length, inputs_embeds.size(2), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                final_attention_mask = torch.zeros(batch_size, max_final_length, dtype=attention_mask.dtype, device=inputs_embeds.device)
                final_input_ids = torch.full((batch_size, max_final_length), fill_value=self.pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
                if labels is not None:
                    final_labels = torch.full((batch_size, max_final_length), fill_value=self.config.ignore_index, dtype=labels.dtype, device=labels.device)
                else:
                    final_labels = None

                for i in range(batch_size):
                    final_input_embeds[i, :before_prompt_lengths[i]] = inputs_embeds_before_prompt[i]
                    final_attention_mask[i, :before_prompt_lengths[i]] = attention_mask_before_prompt[i]
                    final_input_ids[i, :before_prompt_lengths[i]] = input_ids_before_prompt[i]
                    
                    final_input_embeds[i, before_prompt_lengths[i]:before_prompt_lengths[i] + sap_prompt_lengths[i]] = sap_inputs_embeds[i][:sap_prompt_lengths[i]]
                    final_attention_mask[i, before_prompt_lengths[i]:before_prompt_lengths[i] + sap_prompt_lengths[i]] = sap_attention_mask[i][:sap_prompt_lengths[i]]
                    
                    final_input_embeds[i, before_prompt_lengths[i] + sap_prompt_lengths[i]:before_prompt_lengths[i] + sap_prompt_lengths[i] + inputs_embeds_after_prompt[i].size(0)] = inputs_embeds_after_prompt[i]
                    final_attention_mask[i, before_prompt_lengths[i] + sap_prompt_lengths[i]:before_prompt_lengths[i] + sap_prompt_lengths[i] + attention_mask_after_prompt[i].size(0)] = attention_mask_after_prompt[i]
                    final_input_ids[i, before_prompt_lengths[i] + sap_prompt_lengths[i]:before_prompt_lengths[i] + sap_prompt_lengths[i] + input_ids_after_prompt[i].size(0)] = input_ids_after_prompt[i]

                    if labels is not None:
                        final_labels[i, :before_prompt_lengths[i]] = labels_before_prompt[i]
                        final_labels[i, before_prompt_lengths[i]:before_prompt_lengths[i] + sap_prompt_lengths[i]] = torch.full((sap_prompt_lengths[i],), fill_value=-100, dtype=final_labels.dtype, device=final_labels.device)
                        final_labels[i, before_prompt_lengths[i] + sap_prompt_lengths[i]:before_prompt_lengths[i] + sap_prompt_lengths[i] + labels_after_prompt[i].size(0)] = labels_after_prompt[i]

                
                inputs_embeds, attention_mask, labels, position_ids, _ = self._merge_input_ids_with_audio_features(
                    audio_features, audio_output_lengths, final_input_embeds, final_input_ids, final_attention_mask, final_labels
                )                

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2AudioCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            attention_mask=attention_mask,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        input_features=None,
        attention_mask=None,
        **kwargs,
    ):
        # Overwritten -- custom processing (note: might not be needed, but there are no generation tests running atm)

        if past_key_values is not None:
            # if isinstance(past_key_values, Cache):
            #     cache_length = past_key_values.get_seq_length()
            #     past_length = past_key_values.seen_tokens
            # else:
            #     cache_length = past_length = past_key_values[0][0].shape[2]

            # # Here, we get the attention_mask, which was previously stored in the state after _merge_input_ids_with_audio_features.
            # if input_features is not None and kwargs.get("attention_mask") is not None:
            #     attention_mask = kwargs["attention_mask"]
            #     attention_mask = torch.cat(
            #         [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            #     )

            # # Keep only the unprocessed tokens:
            # # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # # input)
            # if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            #     input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # # input_ids based on the past_length.
            # elif past_length < input_ids.shape[1]:
            #     input_ids = input_ids[:, past_length:]
            # # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            # elif self.config.audio_token_index in input_ids:
            #     input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # # older attention values, as their corresponding values are not part of the input.
            # if cache_length < past_length and attention_mask is not None:
            #     attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]
            
            # TODO: consider more complicated scenarios
            # With SAP, sequence length of inputs_embeds is not equal to input_ids after the first forward. Only consider the last token.
            input_ids = input_ids[:, input_ids.shape[1] - 1 :]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        feature_attention_mask = kwargs.get("feature_attention_mask", None)
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "input_features": input_features,
                "feature_attention_mask": feature_attention_mask,
            }
        )
        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        # update past_key_values keeping its naming used in model code
        cache_name, cache = self._extract_past_from_model_output(outputs)
        model_kwargs[cache_name] = cache
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update attention_mask
        if getattr(outputs, "attention_mask", None) is not None:
            model_kwargs["attention_mask"] = outputs.attention_mask

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    dim=-1,
                )

        if model_kwargs.get("use_cache", True):
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
        else:
            past_positions = model_kwargs.pop("cache_position")
            new_positions = torch.arange(
                past_positions[-1] + 1, past_positions[-1] + num_new_tokens + 1, dtype=past_positions.dtype
            ).to(past_positions.device)
            model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))
        return model_kwargs

    def _reorder_cache(self, *args, **kwargs):
        return self.language_model._reorder_cache(*args, **kwargs)
