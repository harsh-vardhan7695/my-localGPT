# my-LocalGPT for Mac

A private, offline document chat system optimized for Mac Mini M4. Chat with your documents using AI without your data ever leaving your computer.

![LocalGPT Interface](https://img.shields.io/badge/Mac%20M4-Optimized-blue)
![Privacy](https://img.shields.io/badge/Privacy-100%25%20Local-green)
![AI Powered](https://img.shields.io/badge/AI-LLaMA%203-orange)

## ✨ Features

- 🔒 **100% Private**: All processing happens locally on your Mac M4
- 🚀 **Mac M4 Optimized**: Leverages Metal Performance Shaders for fast inference
- 📄 **Multiple Formats**: Support for PDF, TXT, DOCX, XLSX, CSV, Markdown
- 🤖 **Powerful AI**: Uses LLaMA 3 and other state-of-the-art models
- 💬 **Chat Interface**: Modern web-based chat with Streamlit
- 📚 **Source Citations**: See exactly which documents informed each answer
- 🎯 **Easy Setup**: Step-by-step installation guide

## 🖥️ System Requirements

### Minimum Requirements
- **Mac Mini M4** with 16GB RAM (recommended)
- **macOS 12.6** or later
- **Python 3.10** or later

## 🚀 Quick Start Guide

### Step 1: Clone and Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd LocalGPT

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Step 2: Install Dependencies

```bash
# Install requirements optimized for Mac M4
pip install -r requirements.txt

# For Mac M4, ensure llama-cpp-python is installed with Metal support
CMAKE_ARGS="-DLLAMA_METAL=on" pip install --force-reinstall --no-cache-dir llama-cpp-python
```

### Step 3: Verify Installation

```bash
# Test system compatibility
python -c "from utils import check_system_compatibility; print(check_system_compatibility())"
```

You should see output showing your Mac M4 capabilities, including MPS availability.

### Step 4: Choose Your Interface

#### Option A: Web Interface (Recommended)
```bash
# Launch the Streamlit web interface
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`

#### Option B: Command Line Interface
```bash
# Process documents
python ingest.py --device_type mps

# Start chat
python run_localGPT.py --device_type mps
```

## 📖 Detailed Setup Instructions

### 1. Environment Setup

For best performance on Mac M4, ensure you're using the native Python installation:

```bash
# Check if you're using native Python
python3 -c "import platform; print(f'Architecture: {platform.machine()}')"
# Should output: Architecture: arm64

# If using Rosetta, install native Python:
# Download from python.org or use Homebrew:
brew install python@3.11
```

### 2. Model Configuration

Edit `constants.py` to choose your preferred model:

```python
# For 16GB RAM (default - good balance)
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_BASENAME = None

# For lower memory usage (8GB RAM)
MODEL_ID = "TheBloke/Llama-2-7B-Chat-GGUF"
MODEL_BASENAME = "llama-2-7b-chat.Q4_K_M.gguf"

# For maximum performance (24GB+ RAM)
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_BASENAME = None
```

### 3. Document Processing

Place your documents in the `SOURCE_DOCUMENTS/` folder:

```bash
# Supported formats
cp ~/Documents/*.pdf SOURCE_DOCUMENTS/
cp ~/Documents/*.txt SOURCE_DOCUMENTS/
cp ~/Documents/*.docx SOURCE_DOCUMENTS/
```

Then process them:

```bash
python ingest.py --device_type mps
```

### 4. First Run

On first startup, the system will download the AI model (may take 10-30 minutes):

```bash
# Start with web interface
streamlit run streamlit_app.py

# Or command line
python run_localGPT.py --device_type mps
```

## 🎛️ Configuration Options

### Device Types
- `mps` - Use Metal Performance Shaders (Mac M4/M3 recommended)
- `cpu` - CPU-only processing (slower but compatible)
- `cuda` - NVIDIA GPU (not applicable for Mac M4)

### Model Types
- `llama3` - LLaMA 3 models (best quality)
- `llama` - LLaMA 2 models (good compatibility)
- `mistral` - Mistral models (fast inference)

### Memory Optimization

For different RAM configurations:

```python
# constants.py settings

# 16GB RAM
N_GPU_LAYERS = 1
N_BATCH = 512
CONTEXT_WINDOW_SIZE = 4096

# 24GB+ RAM
N_GPU_LAYERS = 35
N_BATCH = 1024
CONTEXT_WINDOW_SIZE = 8192
```

## 🖼️ Web Interface Guide

### Main Features

1. **Document Upload**: Drag and drop files or use the file picker
2. **System Monitor**: View RAM usage and compatibility status
3. **Model Recommendations**: Get suggestions based on your hardware
4. **Chat Interface**: Ask questions about your documents
5. **Source Citations**: See which documents informed each answer

### Tips for Best Performance

- Process documents in batches of 10-20 files
- Use GGUF models for lower memory usage
- Monitor RAM usage in the sidebar
- Clear chat history periodically to free memory

## 🔧 Troubleshooting

### Common Issues

#### "MPS not available"
```bash
# Check Metal support
python -c "import torch; print(torch.backends.mps.is_available())"

# If False, use CPU mode:
python run_localGPT.py --device_type cpu
```

#### "Out of memory" errors
1. Reduce batch size in `constants.py`:
   ```python
   N_BATCH = 256  # Instead of 512
   ```
2. Use quantized models:
   ```python
   MODEL_ID = "TheBloke/Llama-2-7B-Chat-GGUF"
   MODEL_BASENAME = "llama-2-7b-chat.Q4_K_M.gguf"
   ```

#### Slow performance
1. Ensure you're using MPS device
2. Check available RAM in Activity Monitor
3. Close other applications
4. Consider using smaller models

#### Model download fails
```bash
# Manual download for GGUF models
mkdir -p models
cd models
# Download from Hugging Face manually
```

### Performance Tips

1. **Use SSD storage** for models and database
2. **Close other applications** while running
3. **Use quantized models** for better performance
4. **Process documents gradually** rather than all at once
5. **Monitor temperature** - Mac M4 may throttle under heavy load

## 📊 Performance Benchmarks

### Mac Mini M4 (16GB RAM)

| Model | RAM Usage | Tokens/sec | Quality |
|-------|-----------|------------|---------|
| LLaMA 3 8B (Full) | ~14GB | 8-12 | Excellent |
| LLaMA 2 7B (Q4) | ~6GB | 15-20 | Very Good |
| Mistral 7B | ~12GB | 10-15 | Good |

## 🤝 Usage Examples

### Command Line
```bash
# Quick start
python run_localGPT.py --device_type mps --model_type llama3

# With history
python run_localGPT.py --device_type mps --use_history

# Show sources
python run_localGPT.py --device_type mps --show_sources
```

### Python API
```python
from run_localGPT import retrieval_qa_pipline

# Initialize QA system
qa = retrieval_qa_pipline(device_type="mps", use_history=False)

# Ask a question
response = qa("What is the main topic of the documents?")
print(response['result'])
```

## 🔒 Privacy and Security

- **No data transmission**: Everything runs locally
- **No cloud dependencies**: Works completely offline after setup
- **Secure storage**: Documents and embeddings stored locally
- **No telemetry**: No usage data collected or transmitted

## 📝 Supported File Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| PDF | `.pdf` | Text and images extracted |
| Text | `.txt` | Plain text files |
| Word | `.docx` | Modern Word documents |
| Excel | `.xlsx` | Spreadsheet data |
| CSV | `.csv` | Comma-separated values |
| Markdown | `.md` | Markdown formatted text |

## 🎯 Advanced Usage

### Custom Embeddings
```python
# In constants.py, change embedding model
EMBEDDING_MODEL_NAME = "intfloat/e5-large-v2"  # Faster alternative
```

### Batch Processing
```bash
# Process multiple document folders
for folder in ~/Documents/*/; do
    cp "$folder"*.pdf SOURCE_DOCUMENTS/
    python ingest.py --device_type mps
done
```

### API Integration
```python
# Create a simple API wrapper
from flask import Flask, request, jsonify
from run_localGPT import retrieval_qa_pipline

app = Flask(__name__)
qa = retrieval_qa_pipline(device_type="mps")

@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.json.get('question')
    response = qa(question)
    return jsonify({'answer': response['result']})
```

## 🤖 Model Recommendations

### For 16GB RAM
1. **LLaMA 3 8B** - Best overall quality
2. **Mistral 7B** - Good balance of speed/quality  
3. **LLaMA 2 7B (Q4)** - Most memory efficient

### For 24GB+ RAM
1. **LLaMA 3 8B** - Full precision, excellent quality
2. **Mixtral 8x7B** - Advanced reasoning (if available)
3. **CodeLlama 13B** - For code-heavy documents


### Getting Help

1. **Check logs**: Look for error messages in terminal output
2. **System info**: Run system compatibility check
3. **Issues**: Check existing GitHub issues
4. **Discord/Forum**: Join community discussions

## 🔄 Updates and Maintenance

### Updating Models
```bash
# Clear model cache
rm -rf ~/.cache/huggingface/transformers/

# Re-download with latest version
python run_localGPT.py --device_type mps
```

### Database Maintenance
```bash
# Clear and rebuild database
rm -rf DB/
python ingest.py --device_type mps
```

## 📄 License

This project idea was taken from PrivateGPT. Please see LICENSE file for details.

## 🙏 Acknowledgments

- PrivateGPT
- LangChain for the RAG framework
- Hugging Face for model hosting
- Meta for LLaMA models

---

For issues or questions, please open a GitHub issue.
