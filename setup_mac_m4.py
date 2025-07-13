#!/usr/bin/env python3
"""
LocalGPT Setup Script for Mac M4
Automates the setup and verification process for optimal Mac M4 performance.
"""

import os
import sys
import subprocess
import platform
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_system_requirements():
    """Check if the system meets requirements for Mac M4 LocalGPT."""
    logger.info("🔍 Checking system requirements...")
    
    # Check platform
    if platform.system() != "Darwin":
        logger.error("❌ This script is designed for macOS only")
        return False
    
    # Check architecture
    if platform.machine() != "arm64":
        logger.warning("⚠️ Not running on Apple Silicon. Performance may be limited.")
    else:
        logger.info("✅ Running on Apple Silicon (arm64)")
    
    # Check Python version
    py_version = sys.version_info
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 10):
        logger.error(f"❌ Python 3.10+ required. Found {py_version.major}.{py_version.minor}")
        return False
    else:
        logger.info(f"✅ Python {py_version.major}.{py_version.minor} detected")
    
    # Check macOS version
    macos_version = platform.mac_ver()[0]
    logger.info(f"✅ macOS {macos_version} detected")
    
    return True

def setup_directories():
    """Create necessary directories."""
    logger.info("📁 Setting up directories...")
    
    directories = [
        "SOURCE_DOCUMENTS",
        "DB", 
        "models",
        ".logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"✅ Created directory: {directory}")

def install_dependencies():
    """Install Python dependencies optimized for Mac M4."""
    logger.info("📦 Installing Python dependencies...")
    
    try:
        # Upgrade pip first
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True)
        logger.info("✅ Pip upgraded")
        
        # Install basic requirements
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        logger.info("✅ Basic requirements installed")
        
        # Install llama-cpp-python with Metal support for Mac M4
        logger.info("🔧 Installing llama-cpp-python with Metal support...")
        env = os.environ.copy()
        env["CMAKE_ARGS"] = "-DLLAMA_METAL=on"
        
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "--force-reinstall", "--no-cache-dir", "llama-cpp-python"
        ], env=env, check=True)
        logger.info("✅ llama-cpp-python with Metal support installed")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install dependencies: {e}")
        return False
    
    return True

def verify_torch_mps():
    """Verify PyTorch MPS (Metal Performance Shaders) support."""
    logger.info("🔧 Verifying PyTorch MPS support...")
    
    try:
        import torch
        
        logger.info(f"✅ PyTorch {torch.__version__} installed")
        
        if torch.backends.mps.is_available():
            logger.info("✅ Metal Performance Shaders (MPS) available")
            return True
        else:
            logger.warning("⚠️ MPS not available. Will fall back to CPU.")
            return False
            
    except ImportError:
        logger.error("❌ PyTorch not installed")
        return False

def test_imports():
    """Test if all required modules can be imported."""
    logger.info("🧪 Testing module imports...")
    
    modules_to_test = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("langchain", "LangChain"),
        ("streamlit", "Streamlit"),
        ("chromadb", "ChromaDB"),
        ("sentence_transformers", "Sentence Transformers"),
    ]
    
    success = True
    for module, name in modules_to_test:
        try:
            __import__(module)
            logger.info(f"✅ {name} import successful")
        except ImportError as e:
            logger.error(f"❌ {name} import failed: {e}")
            success = False
    
    return success

def create_sample_documents():
    """Create sample documents for testing."""
    logger.info("📝 Creating sample documents...")
    
    sample_docs = {
        "mac_m4_overview.txt": """Mac Mini M4 Overview

The Mac mini M4 is Apple's latest compact desktop computer featuring the M4 chip.

Key Features:
- M4 chip with 8-core CPU and 10-core GPU
- Available with 16GB, 24GB, or 32GB unified memory
- 256GB, 512GB, 1TB, or 2TB SSD storage options
- Compact 7.7-inch square design
- Multiple ports including Thunderbolt 4, USB-A, HDMI, and Ethernet

The M4 chip provides excellent performance for AI workloads and machine learning tasks,
making it ideal for running large language models locally.

Performance Benefits:
- Metal Performance Shaders acceleration
- Unified memory architecture
- Low power consumption
- Excellent thermal management

The Mac mini M4 is perfect for developers, content creators, and anyone needing
a powerful yet compact desktop computer.
""",
        
        "localgpt_guide.md": """# LocalGPT on Mac M4

## Introduction
LocalGPT allows you to run AI chatbots completely locally on your Mac M4,
ensuring complete privacy and data security.

## Benefits
- Complete privacy - no data leaves your computer
- Fast inference with Metal Performance Shaders
- Support for multiple document formats
- Offline operation after initial setup

## Supported Models
- LLaMA 3 8B (recommended for 16GB+ RAM)
- Mistral 7B (good balance of speed and quality)
- LLaMA 2 7B (efficient for lower memory)

## Performance Tips
- Use GGUF quantized models for better memory efficiency
- Enable Metal acceleration for faster inference
- Monitor RAM usage during operation
- Process documents in batches for better performance

## Troubleshooting
If you encounter issues:
1. Check available memory
2. Verify MPS support
3. Use smaller batch sizes
4. Consider quantized models
""",
        
        "ai_privacy.txt": """AI Privacy and Local Computing

In an era of increasing data privacy concerns, running AI models locally
has become increasingly important for individuals and organizations.

Benefits of Local AI:
- Data Privacy: Your documents never leave your device
- No Internet Required: Works completely offline
- No Subscriptions: One-time setup, no recurring costs
- Full Control: You control the model and data processing

Challenges:
- Hardware Requirements: Needs sufficient RAM and processing power
- Model Downloads: Initial setup requires downloading large models
- Technical Setup: More complex than cloud-based solutions

The Mac M4 with its unified memory architecture and Metal Performance Shaders
provides an excellent platform for local AI processing, offering the perfect
balance of performance, efficiency, and privacy.

Security Considerations:
- Models run in isolation
- No network access required during inference
- Local storage of all data and embeddings
- Full control over data retention and deletion
"""
    }
    
    for filename, content in sample_docs.items():
        file_path = Path("SOURCE_DOCUMENTS") / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"✅ Created sample document: {filename}")

def run_system_check():
    """Run a comprehensive system check."""
    logger.info("🔍 Running system compatibility check...")
    
    try:
        from utils import check_system_compatibility, log_system_info
        
        # Log system information
        system_info = log_system_info()
        
        # Check memory requirements
        memory_gb = system_info.get('total_memory_gb', 0)
        if memory_gb >= 16:
            logger.info(f"✅ Memory: {memory_gb:.1f}GB (Sufficient for most models)")
        elif memory_gb >= 8:
            logger.warning(f"⚠️ Memory: {memory_gb:.1f}GB (Use quantized models)")
        else:
            logger.error(f"❌ Memory: {memory_gb:.1f}GB (Insufficient)")
            return False
        
        # Check MPS availability
        if system_info.get('mps_available', False):
            logger.info("✅ Metal Performance Shaders available")
        else:
            logger.warning("⚠️ MPS not available, will use CPU")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Could not import utils module: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 LocalGPT Mac M4 Setup Script")
    print("=" * 40)
    
    # Check system requirements
    if not check_system_requirements():
        logger.error("❌ System requirements not met")
        sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    # Install dependencies
    if not install_dependencies():
        logger.error("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Verify PyTorch MPS
    verify_torch_mps()
    
    # Test imports
    if not test_imports():
        logger.error("❌ Module import tests failed")
        sys.exit(1)
    
    # Create sample documents
    create_sample_documents()
    
    # Run system check
    if not run_system_check():
        logger.warning("⚠️ System check completed with warnings")
    
    print("\n" + "=" * 40)
    print("✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Process sample documents: python ingest.py --device_type mps")
    print("2. Start web interface: streamlit run streamlit_app.py")
    print("3. Or use CLI: python run_localGPT.py --device_type mps")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main() 