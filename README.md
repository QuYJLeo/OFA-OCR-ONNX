# OFA-OCR-ONNX

A powerful Optical Character Recognition (OCR) system based on the **OFA (One For All)** model, optimized for efficient inference using ONNX runtime. This project provides high-performance text recognition capabilities with support for Chinese and English text.

## 🌟 Features

- **State-of-the-art OCR**: Built on the OFA unified multimodal model, achieving exceptional accuracy on text recognition tasks
- **ONNX Optimization**: Leverages ONNX Runtime for fast, cross-platform inference on CPU
- **Chinese & English Support**: Native support for both Chinese and English text recognition
- **Beam Search Decoding**: Implements efficient beam search algorithm for optimal sequence generation
- **Lightweight & Portable**: No heavy framework dependencies, easy to deploy
- **Pre-trained Models**: Includes pre-quantized ONNX models for immediate use

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy onnxruntime opencv-python
```

### Installation

```bash
git clone https://github.com/your-repo/OFA-OCR-ONNX.git
cd OFA-OCR-ONNX
```

### Download Model

Download the ONNX model files from Google Drive and place them in the `onxw/` directory:

[Model Files](https://drive.google.com/drive/folders/1SM4d0P_M5Km3UEca61pBWGOIvNKFw5jH)

### Usage

```python
from oxn_infer import OfaTasks

# Initialize the OCR engine
ocr = OfaTasks()

# Perform OCR on an image
result = ocr("path/to/your/image.png")
print("Recognized text:", result)
```

### Command Line Interface

```bash
python oxn_infer.py
```

## 📁 Project Structure

```
OFA-OCR-ONNX/
├── onxw/                    # ONNX model files
│   ├── encoder.onnx         # Full precision encoder
│   ├── encoder_qu.onnx      # Quantized encoder (recommended)
│   ├── decoder.onnx         # Full precision decoder
│   ├── decoder_qu.onnx      # Quantized decoder (recommended)
│   └── vocab.txt            # Vocabulary file
├── utils/                   # Utility modules
│   ├── preprocess.py        # Image preprocessing pipeline
│   ├── tokenization.py      # Tokenizer for Chinese text
│   └── 优化模型文件.py       # Model optimization utilities
├── te/                      # Test images directory
│   ├── ff.png
│   └── yk.png
├── oxn_infer.py             # Main inference script (FP32)
├── oxn_infer_f16.py         # Inference script (FP16)
├── LICENSE                  # License file
└── README.md                # This file
```

## 🧠 Model Architecture

### Encoder
- **Input**: RGB images of size 480x480
- **Output**: Hidden states, attention masks, and position embeddings
- **Layers**: 12 transformer encoder layers
- **Hidden Size**: 768

### Decoder
- **Input**: Encoder outputs + token sequences
- **Output**: Logits for token prediction
- **Layers**: 12 transformer decoder layers
- **Beam Size**: 5 (configurable)

### Preprocessing Pipeline
1. **Resize**: Maintain aspect ratio, pad to 480x480
2. **Normalize**: Scale to [-1, 1] with mean [0.5, 0.5, 0.5]
3. **Format**: Convert to CHW format for model input

## ⚡ Performance

| Metric | Value |
|--------|-------|
| Encoder Time | ~200ms (CPU) |
| Decoder Time | ~300ms (CPU) |
| Total Inference | ~500ms per image |
| Supported Languages | Chinese, English |
| Max Text Length | 17 tokens |

## 📊 Example Results

### Input Image

![Test Image 1](te/ff.png)

### Output
```
Result: ["刘亦菲"]
```

## 🔧 Configuration

You can customize the following parameters in `oxn_infer.py`:

- **Beam Size**: Adjust `active_hypos` for trade-off between speed and accuracy
- **Max Length**: Modify the `range(17)` for longer text sequences
- **Quantization**: Switch between quantized (`_qu.onnx`) and full precision models
- **Execution Provider**: Change `providers` to use GPU (`["CUDAExecutionProvider"]`)

## 🛠️ Development

### Adding New Languages

1. Add vocabulary to `onxw/vocab.txt`
2. Update `utils/tokenization.py` for new token types
3. Fine-tune or convert model with new language data

### Model Optimization

Run the optimization script to quantize your own models:

```bash
python utils/优化模型文件.py
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OFA: Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework](https://arxiv.org/abs/2202.03052)
- [ONNX Runtime](https://onnxruntime.ai/) for high-performance inference
- [OpenCV](https://opencv.org/) for image processing

## 📬 Contact

For questions, issues, or contributions, please open an issue on the repository.

---

*Made with ❤️ for the OCR community*