# SAP²-ASR: 面向上下文感知自动语音识别的语音感知长上下文剪枝与集成

> **注意**：本仓库是基于 [ms-swift](https://github.com/modelscope/ms-swift) 的 fork，实现了 SAP²（Speech-Aware Context Pruning with Speech-Driven Attention-based Pooling，语音感知上下文剪枝与语音驱动注意力池化）方法，用于上下文感知的自动语音识别，详见我们的[论文](https://www.arxiv.org/pdf/2511.11139)。

<p align="center">
    <br>
    <img src="asset/banner.png"/>
    <br>
<p>
<p align="center">
<a href="https://www.arxiv.org/pdf/2511.11139">论文</a> &nbsp ｜ &nbsp <a href="https://github.com/jymh/SAP2-ASR">原始代码</a> 
<br>
        <a href="README_CN.md">中文</a> &nbsp ｜ &nbsp <a href="README.md">English</a> &nbsp
</p>

<p align="center">
<img src="https://img.shields.io/badge/python-3.10-5be.svg">
<img src="https://img.shields.io/badge/pytorch-%E2%89%A52.0-orange.svg">
<a href="https://github.com/modelscope/swift/blob/main/LICENSE"><img src="https://img.shields.io/github/license/modelscope/swift"></a>
</p>

## 📖 目录
- [简介](#-简介)
- [安装](#%EF%B8%8F-安装)
- [数据集](#-数据集)
- [快速开始](#-快速开始)
- [使用说明](#-使用说明)
- [模型架构](#-模型架构)
- [引用](#-引用)
- [许可证](#-许可证)

## 📝 简介

**SAP²（Speech-Aware Context Pruning with Speech-Driven Attention-based Pooling，语音感知上下文剪枝与语音驱动注意力池化）** 是一个用于上下文感知自动语音识别（ASR）的新框架，能够动态剪枝并集成相关的上下文关键词。该方法解决了在特定领域场景（如会议演讲）中利用长上下文信息的挑战，这些场景中大量来自OCR的文本上下文既包含相关信息，也包含大量噪声。

### 核心特性

- **语音感知上下文剪枝**：动态过滤来自OCR的文本上下文，仅保留与语音内容直接相关的关键词
- **跨模态上下文压缩**：使用语音驱动注意力池化（Speech-Driven Attention-based Pooling）将大量文本输入压缩为简洁的、与语音相关的上下文嵌入
- **最先进的性能**：在 SlideSpeech 数据集上达到 7.71% 的词错误率（WER），在 LibriSpeech 数据集上达到 1.12% 的 WER，相比非上下文基线，在偏向关键词识别方面相对提升了 41.1%

### 实验结果

- **SlideSpeech**：WER 7.71%，B-WER 相比基线提升 41.1%
- **LibriSpeech**：WER 1.12%
- 在大量上下文输入条件下具有**鲁棒的可扩展性**

### 识别示例

下图展示了 SAP² 与之前方法在 SlideSpeech 测试集上的识别示例对比。红色文本表示专有名词的识别错误，绿色高亮文本展示了 SAP² 所做的修正。

<p align="center">
  <img src="asset/figure1.jpg" alt="识别示例" width="800"/>
</p>

## 🛠️ 安装

本项目基于 [ms-swift](https://github.com/modelscope/ms-swift)。安装方法如下：

```shell
# 克隆仓库
git clone https://github.com/jymh/SAP2-ASR.git
cd SAP2-ASR

# 创建 conda 环境
conda env create -f environment.yml

# 激活环境
conda activate swift

# 安装包
pip install -e .
```

**环境要求：**
- Python >= 3.10
- PyTorch >= 2.0
- transformers >= 4.45
- librosa（用于音频处理）

## 📊 数据集

本项目使用两个数据集进行评估：**SlideSpeech** 和 **LibriSpeech**。两个数据集都可以在 OpenSLR 找到，或者您可以从以下来源下载：

### SlideSpeech

SlideSpeech 是一个包含幻灯片的大规模音视频语料库，包含 1,705 个视频，超过 1,000 小时的音频，其中包括 473 小时的高质量转录语音。

**下载方式：**
1. **GitHub 仓库**：从 [https://github.com/Mashiro009/slidespeech_dl.git](https://github.com/Mashiro009/slidespeech_dl.git) 克隆官方下载脚本
   ```shell
   git clone https://github.com/Mashiro009/slidespeech_dl.git
   cd slidespeech_dl
   bash run.sh
   ```

2. **OpenSLR**：可在 OpenSLR 网站获取

**数据集详情：**
- 网站：[https://slidespeech.github.io/](https://slidespeech.github.io/)
- 包含同步的幻灯片和 OCR 提取的文本上下文
- 适用于上下文感知 ASR 评估

### LibriSpeech

LibriSpeech 是一个大规模英语朗读语音语料库，源自 LibriVox 项目的有声读物。

**下载方式：**
1. **Hugging Face Datasets**：使用 Hugging Face datasets 库直接加载
   ```python
   from datasets import load_dataset
   dataset = load_dataset("openslr/librispeech_asr")
   ```
   或访问：[https://huggingface.co/datasets/openslr/librispeech_asr](https://huggingface.co/datasets/openslr/librispeech_asr)

2. **OpenSLR**：可在 OpenSLR 网站获取

**数据集详情：**
- 包含约 1000 小时的 16kHz 英语朗读语音
- 分为训练集（train-clean、train-other）、验证集和测试集
- 广泛用于 ASR 系统基准测试

**注意**：对于 LibriSpeech，我们遵循论文中的方法，为训练集和验证集动态构建偏置列表，使用 common5k 词汇表之外的单词和随机选择的干扰词。

### 预处理数据集元数据

我们在 Hugging Face 上提供了预处理好的数据集元数据，包含为 SAP² 格式化的上下文关键词训练数据。元数据包含来自 SlideSpeech 和 LibriSpeech 数据集的 109 万训练样本。

**Hugging Face 数据集**：[https://huggingface.co/datasets/jymh/SAP2-ASR](https://huggingface.co/datasets/jymh/SAP2-ASR)

## 🚀 快速开始

### 使用 SAP（语音驱动注意力池化）训练 SAP² 模型

以下示例展示如何在 SlideSpeech 数据集上使用 SAP 池化训练 SAP² 模型：

```shell
# 使用 SAP 压缩进行多 GPU 训练
NPROC_PER_NODE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 swift sft \
    --model "/path/to/qwen2-audio-instruct" \
    --model_type sap_qwen2_audio \
    --dataset "/path/to/slidespeech/train.json" \
    --val_dataset "/path/to/slidespeech/dev.json" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --max_length 4096 \
    --output_dir "/path/to/output" \
    --train_type lora \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner false \
    --lora_rank 8 \
    --sap_window_size 2 \
    --compressor_hidden_size 4096 \
    --num_attention_heads 4 \
    --deepspeed zero2
```

**关键参数：**
- `--model_type sap_qwen2_audio`：使用支持 SAP 的 Qwen2-Audio 模型
- `--sap_window_size 2`：语音驱动注意力池化的窗口大小
- `--compressor_hidden_size 4096`：压缩器的隐藏层大小
- `--num_attention_heads 4`：池化使用的注意力头数量

### 使用 SAP² 模型进行推理

训练完成后，使用训练好的模型进行推理：

```shell
CUDA_VISIBLE_DEVICES=0 swift infer \
    --adapters /path/to/checkpoint-xxx \
    --infer_backend pt \
    --temperature 0 \
    --max_batch_size 4 \
    --val_dataset /path/to/test.json \
    --result_path /path/to/result.jsonl \
    --stream false \
    --sap_window_size 2 \
    --compressor_hidden_size 4096 \
    --num_attention_heads 4
```

## ✨ 使用说明

### 数据准备

SAP² 方法要求上下文关键词（例如来自 OCR 文本）使用特殊标记 `<|startofcontext|>` 和 `<|endofcontext|>` 进行格式化。数据格式示例：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "<audio>/path/to/audio.wav</audio>Transcribe speech to text according to keywords may appear in the utterance. Possible keywords are: <|startofcontext|>keyword1 keyword2 keyword3<|endofcontext|>"
    },
    {
      "role": "assistant",
      "content": "transcribed text"
    }
  ],
  "audios": "/path/to/audio.wav"
}
```

您可以使用 `extract_predicted_keywords.py` 处理数据并添加上下文关键词。

### 使用 SAP 压缩进行训练

SAP（语音驱动注意力池化）机制使用语音驱动注意力池化压缩长上下文关键词：

```shell
swift sft \
    --model_type sap_qwen2_audio \
    --model "/path/to/qwen2-audio-instruct" \
    --dataset "/path/to/dataset" \
    --train_type lora \
    --sap_window_size 2 \
    --compressor_hidden_size 4096 \
    --num_attention_heads 4 \
    ...
```

### 评估

推理完成后，您可以使用提供的评估脚本评估结果：

```shell
python evaluate_slidespeech_process.py --input_file /path/to/result.jsonl
```

## 🏗️ 模型架构

下图展示了 SAP² 的整体架构：

<p align="center">
  <img src="asset/main_fig.jpg" alt="SAP² 模型架构" width="800"/>
</p>

核心实现位于 `swift/llm/model/sqp_models/modeling_sqp_qwen2audio.py`，扩展了 `Qwen2AudioForConditionalGeneration`，包含：

- **`Qwen2AudioSAPPoolingLayer`**：实现 SAP（语音驱动注意力池化），基于语音特征压缩上下文关键词
- **`SAP2Qwen2AudioForConditionalGeneration`**：将 SAP 压缩集成到 Qwen2-Audio 架构中的主模型类

SAP 池化层使用语音嵌入和上下文嵌入之间的跨模态注意力来计算池化权重，能够高效压缩长上下文输入，同时保留与语音相关的信息。

## 📎 引用

如果您在研究中使用了 SAP²，请引用我们的论文：

```bibtex
@article{rong2025speechaware,
  title={Speech-Aware Long Context Pruning and Integration for Contextualized Automatic Speech Recognition},
  author={Rong, Yiming and Zhang, Yixin and Wang, Ziyi and Jiang, Deyang and Zhao, Yunlong and Wu, Haoran and Zhou, Shiyu and Xu, Bo},
  journal={arXiv preprint arXiv:2511.11139},
  year={2025}
}
```


## 🏛 许可证

本框架使用 [Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE) 进行许可。模型和数据集请查看原资源页面并遵守对应的许可证。
