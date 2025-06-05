# Qwen2.5-VL Video GRPO Training

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-VL/qwen2.5vl_logo.png" width="400"/>
<p>

<p align="center">
        💜 <a href="https://chat.qwenlm.ai/"><b>Qwen Chat</b></a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/qwen">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://qwenlm.github.io/blog/qwen2.5-vl/">Blog</a>&nbsp&nbsp | &nbsp&nbsp📚 <a href="https://github.com/QwenLM/Qwen2.5-VL/tree/main/cookbooks">Cookbooks</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://arxiv.org/abs/2502.13923">Paper</a>&nbsp&nbsp
<br>
🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen2.5-VL-72B-Instruct">Demo</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://github.com/QwenLM/Qwen/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp | &nbsp&nbsp🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://help.aliyun.com/zh/model-studio/developer-reference/qwen-vl-api">API</a>&nbsp&nbsp | &nbsp&nbsp🖥️ <a href="https://gallery.pai-ml.com/#/preview/deepLearning/cv/qwen2.5-vl">PAI-DSW</a>
</p>

## 🎯 Project Overview

This repository provides **Video GRPO (Group Relative Policy Optimization)** training capabilities for Qwen2.5-VL models, specifically designed for **multiple-choice video understanding tasks**. While based on the original Qwen2.5-VL codebase, this project focuses on implementing advanced preference optimization techniques for video comprehension.

## 🚀 Key Features & New Implementations

### 🎬 Video GRPO Training
- **Group Relative Policy Optimization**: Advanced preference learning for multiple-choice video questions
- **Comprehensive Metrics Logging**: Detailed wandb integration with policy probabilities, KL divergence, and reward tracking
- **Format-Aware Training**: Automatic detection and reward for proper response formatting (`<think>...</think> <answer>...</answer>`)
- **Multi-Choice Optimization**: Specialized handling of video understanding tasks with multiple answer options

### 🔧 New Components Added
1. **GRPO Trainer** (`qwenvl/train/grpo_trainer.py`): Custom trainer implementing group-wise preference optimization
2. **GRPO Dataset Class** (`qwenvl/data/data_qwen_grpo.py`): Specialized data loader for multiple-choice video tasks
3. **GRPO Training Script** (`qwenvl/train/train_grpo.py`): Main training entry point with GRPO-specific configurations
4. **Comprehensive Logging**: Real-time metrics tracking for training optimization
5. **Ready-to-Use Scripts**: Shell scripts for single and multi-GPU GRPO training

### 📊 GRPO Metrics Dashboard
Track detailed training metrics in real-time:
- **Policy Probabilities**: `grpo/policy_seq_log_prob`
- **KL Divergence**: `grpo/kl_div` 
- **Format Rewards**: `grpo/format_reward`
- **Accuracy Rewards**: `grpo/accuracy_reward`
- **Total Rewards**: `grpo/total_reward`
- **Advantage Values**: `grpo/advantage`

## 🎯 Use Cases

This implementation is ideal for:
- **Video Question Answering**: Training models to answer multiple-choice questions about video content
- **Video Reasoning Tasks**: Improving logical reasoning capabilities for video understanding
- **Preference Learning**: Optimizing model responses based on correctness and format preferences
- **Research Applications**: Exploring advanced preference optimization techniques for multimodal models

## 🛠️ Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Qwen2.5-VL-Video-GRPO.git
cd Qwen2.5-VL-Video-GRPO

# Install dependencies
pip install -r requirements.txt
pip install qwen-vl-utils[decord]

# Install specific versions for best compatibility
pip install transformers==4.51.3 accelerate
```

### Basic GRPO Training

```bash
# Navigate to the training directory
cd qwen-vl-finetune

# Make training script executable
chmod +x scripts/train_grpo.sh

# Run GRPO training
./scripts/train_grpo.sh
```

### Custom GRPO Training

```bash
python qwenvl/train/train_grpo.py \
    --model_name_or_path "Qwen/Qwen2.5-VL-3B-Instruct" \
    --dataset_use "your_video_dataset_grpo" \
    --output_dir "./output/grpo_model" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --grpo_alpha 0.5 \
    --grpo_beta 0.1 \
    --format_reward_weight 0.3 \
    --accuracy_reward_weight 0.7 \
    --grpo_sample_size 4 \
    --report_to wandb \
    --run_name "video-grpo-experiment"
```

## 📚 Documentation

### GRPO Training Guide
- [Detailed GRPO Implementation](qwen-vl-finetune/GRPO_README.md)
- [Training Framework Documentation](qwen-vl-finetune/README.md)
- [Dataset Preparation Guide](#dataset-format)

### Key GRPO Parameters
- `grpo_alpha`: Reward scaling factor (default: 0.5)
- `grpo_beta`: KL penalty weight (default: 0.1)
- `format_reward_weight`: Weight for format compliance (default: 0.3)
- `accuracy_reward_weight`: Weight for answer accuracy (default: 0.7)
- `grpo_sample_size`: Number of samples per example (default: 4)
- `grpo_logging_steps`: Metrics logging frequency (default: 50)

## 📊 Dataset Format

Your video dataset should contain multiple-choice questions:

```json
{
    "video": "videos/sample.mp4",
    "conversations": [
        {
            "from": "human",
            "value": "<video>\nWhat caused the blue object to move?\nOptions:\n(A) Gravity\n(B) Collision\n(C) Magnetic force"
        },
        {
            "from": "gpt",
            "value": "<think>I can see in the video that there's a red ball that rolls into the blue object, causing it to move. This is clearly a collision scenario.</think> <answer>(B) Collision</answer>"
        }
    ]
}
```

## 🔄 Format Types

### Thinking Format (Default)
- **Dataset**: Any name not containing "no_think"
- **Expected Output**: `<think>...</think> <answer>...</answer>`
- **Use Case**: When you want the model to show reasoning

### No-Thinking Format
- **Dataset**: Names containing "no_think"
- **Expected Output**: `<answer>...</answer>` only
- **Use Case**: When you want concise answers

## 📈 Monitoring Training

The GRPO trainer provides comprehensive monitoring:
- Real-time loss tracking
- Format compliance metrics
- Answer accuracy measurements
- KL divergence from reference model
- Detailed reward breakdowns

## 🏆 Advantages over Standard Fine-tuning

- **Better Preference Learning**: Direct optimization for choosing correct answers
- **Robust Training**: Less prone to distribution collapse
- **Format Enforcement**: Built-in rewards for proper response structure
- **Detailed Monitoring**: Comprehensive metrics for training insights
- **Video-Specific**: Optimized for video understanding tasks

## 💡 Tips for Best Results

1. **Start Small**: Use small batch sizes (1-2) due to generation overhead
2. **Sample Diversity**: 4-8 samples per example provides good balance
3. **Learning Rate**: Use lower rates (2e-5) for stability
4. **Reward Balance**: Adjust format vs accuracy weights based on priorities
5. **KL Control**: Increase `grpo_beta` if model diverges too much

## 🔧 Troubleshooting

### Out of Memory
- Reduce `grpo_sample_size`
- Reduce `generation_max_length`
- Enable gradient checkpointing
- Use smaller batch sizes

### Poor Generation Quality
- Verify flash attention configuration
- Check tokenizer chat template
- Ensure proper video frame sampling

### Low Rewards
- Verify dataset format matches expectations
- Check ground truth answer formatting
- Adjust reward weights

## 🐳 Docker Support

```bash
# Use the provided Docker image
docker run --gpus all --ipc=host --network=host --rm --name qwen2.5-grpo -it qwenllm/qwenvl:2.5-cu121 bash

# Or build your own
docker build -f Dockerfile -t qwen2.5-vl-grpo .
```

## 📖 Original Qwen2.5-VL Information

This project is built upon the excellent Qwen2.5-VL foundation. Below are the key capabilities inherited from the original model:

### Key Enhancements from Qwen2.5-VL:

* **Powerful Document Parsing Capabilities**: Upgrade text recognition to omnidocument parsing, excelling in processing multi-scene, multilingual, and various built-in (handwriting, tables, charts, chemical formulas, and music sheets) documents.

* **Precise Object Grounding Across Formats**: Unlock improved accuracy in detecting, pointing, and counting objects, accommodating absolute coordinate and JSON formats for advanced spatial reasoning.

* **Ultra-long Video Understanding and Fine-grained Video Grounding**: Extend native dynamic resolution to the temporal dimension, enhancing the ability to understand videos lasting hours while extracting event segments in seconds.

* **Enhanced Agent Functionality for Computer and Mobile Devices**: Leverage advanced grounding, reasoning, and decision-making abilities, boosting the model with superior agent functionality on smartphones and computers.

### Model Sizes Available:
- **Qwen2.5-VL-3B-Instruct**: Compact model for efficient deployment
- **Qwen2.5-VL-7B-Instruct**: Balanced performance and efficiency
- **Qwen2.5-VL-32B-Instruct**: Enhanced capabilities with improved alignment
- **Qwen2.5-VL-72B-Instruct**: Flagship model with maximum performance

### Performance Highlights

| Dataset            | Qwen2.5-VL-3B | Qwen2.5-VL-7B | Qwen2.5-VL-32B | Qwen2.5-VL-72B | GPT-4o | Claude3.5 Sonnet |
|--------------------|---------------|---------------|----------------|----------------|---------|------------------|
| MMMU               | 53.1          | 58.6          | 70.0           | **70.2**       | 70.3    | 70.4             |
| VideoMME           | 61.5/67.6     | 65.1/71.6     | 70.5/77.9      | **73.3/79.1**  | 71.9/77.2| 60/62.9         |
| DocVQA             | 93.9          | 95.7          | 94.8           | **96.4**       | 91.1    | 95.2             |
| MathVista          | 62.3          | 68.2          | 74.7           | **74.8**       | 63.8    | 65.4             |

## 🎯 Cookbooks & Examples

We provide comprehensive examples for various capabilities:

| Cookbook | Description | Open |
| -------- | ----------- | ---- |
| [Universal Recognition](https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/universal_recognition.ipynb) | Identify animals, plants, people, and objects | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QwenLM/Qwen2.5-VL/blob/main/cookbooks/universal_recognition.ipynb) |
| [Document Parsing](https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/document_parsing.ipynb) | Advanced document parsing with layout information | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QwenLM/Qwen2.5-VL/blob/main/cookbooks/document_parsing.ipynb) |
| [Video Understanding](https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/video_understanding.ipynb) | Video OCR, long video understanding, and grounding | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QwenLM/Qwen2.5-VL/blob/main/cookbooks/video_understanding.ipynb) |
| [Mobile Agent](https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/mobile_agent.ipynb) | Mobile phone control and interaction | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QwenLM/Qwen2.5-VL/blob/main/cookbooks/mobile_agent.ipynb) |

## 🚀 Using Base Qwen2.5-VL Models

### Basic Inference

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Load model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# Process input
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "path/to/video.mp4",
                "fps": 1.0,
            },
            {"type": "text", "text": "Describe this video."},
        ],
    }
]

# Generate response
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
inputs = inputs.to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(output_text)
```

### Video Processing Configuration

```python
# Configure video processing parameters
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "https://example.com/video.mp4",
                "min_pixels": 4 * 28 * 28,
                "max_pixels": 256 * 28 * 28,
                "total_pixels": 20480 * 28 * 28,
                "fps": 1.0,
            },
            {"type": "text", "text": "What happens in this video?"},
        ],
    }
]
```

## 🔬 Research & Development

This project extends the capabilities of Qwen2.5-VL for research in:
- **Preference Learning**: Advanced techniques for aligning model outputs with human preferences
- **Video Understanding**: Specialized training for temporal visual reasoning
- **Multi-Modal Reasoning**: Integration of visual and textual information processing
- **Reward Modeling**: Sophisticated reward functions for training optimization

## 📄 Citation

If you use this Video GRPO implementation, please cite both this work and the original Qwen2.5-VL:


## 🤝 Contributing

We welcome contributions to improve the Video GRPO implementation:
1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Submit a pull request

## 📝 License

This project inherits the license from the original Qwen2.5-VL repository. Please see the LICENSE file for details.

## 🆘 Support

For questions about:
- **GRPO Implementation**: Open an issue in this repository
- **Base Qwen2.5-VL**: Refer to the [original repository](https://github.com/QwenLM/Qwen2.5-VL)
- **General Discussion**: Join the [Discord community](https://discord.gg/CV4E9rpNSD)

---

**Note**: This is a specialized implementation focusing on Video GRPO training. For general Qwen2.5-VL usage, please refer to the [original repository](https://github.com/QwenLM/Qwen2.5-VL).




```BibTeX

@article{Qwen2.5-VL,
  title={Qwen2.5-VL Technical Report},
  author={Bai, Shuai and Chen, Keqin and Liu, Xuejing and Wang, Jialin and Ge, Wenbin and Song, Sibo and Dang, Kai and Wang, Peng and Wang, Shijie and Tang, Jun and Zhong, Humen and Zhu, Yuanzhi and Yang, Mingkun and Li, Zhaohai and Wan, Jianqiang and Wang, Pengfei and Ding, Wei and Fu, Zheren and Xu, Yiheng and Ye, Jiabo and Zhang, Xi and Xie, Tianbao and Cheng, Zesen and Zhang, Hang and Yang, Zhibo and Xu, Haiyang and Lin, Junyang},
  journal={arXiv preprint arXiv:2502.13923},
  year={2025}
}

@article{Qwen2-VL,
  title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution},
  author={Wang, Peng and Bai, Shuai and Tan, Sinan and Wang, Shijie and Fan, Zhihao and Bai, Jinze and Chen, Keqin and Liu, Xuejing and Wang, Jialin and Ge, Wenbin and Fan, Yang and Dang, Kai and Du, Mengfei and Ren, Xuancheng and Men, Rui and Liu, Dayiheng and Zhou, Chang and Zhou, Jingren and Lin, Junyang},
  journal={arXiv preprint arXiv:2409.12191},
  year={2024}
}

@article{Qwen-VL,
  title={Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond},
  author={Bai, Jinze and Bai, Shuai and Yang, Shusheng and Wang, Shijie and Tan, Sinan and Wang, Peng and Lin, Junyang and Zhou, Chang and Zhou, Jingren},
  journal={arXiv preprint arXiv:2308.12966},
  year={2023}
}
```

<br>
