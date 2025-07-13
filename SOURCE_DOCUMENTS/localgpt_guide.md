# LocalGPT on Mac M4

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
