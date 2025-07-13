import logging
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,LlamaForCausalLM,LlamaTokenizer
from langchain.llms import LlamaCpp
from transformers import pipeline
import platform
from huggingface_hub import hf_hub_download
from constants import MODELS_PATH,MODEL_ID,MODEL_BASENAME,N_GPU_LAYERS,N_BATCH,CONTEXT_WINDOW_SIZE,MAX_NEW_TOKENS


def load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING):
    """
    Load a GGUF/GGML quantized model using llama-cpp-python.
    Optimized for Mac M4 with Metal Performance Shaders.
    """
    try:
        from llama_cpp import Llama
        
        LOGGING.info(f"Loading GGUF/GGML Model: {model_id}")
        LOGGING.info(f"Model Basename: {model_basename}")
        
        model_path = os.path.join("./models", model_basename)
        
        # Check if model file exists
        if not os.path.exists(model_path):
            LOGGING.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Configure parameters for Mac M4
        if device_type == "mps":
            # Use Metal Performance Shaders for Mac M4
            llm = LlamaCpp(
                model_path=model_path,
                max_tokens=2048,
                n_ctx=4096,
                n_batch=512,
                callback_manager=None,
                verbose=False,
                n_gpu_layers=1,  # Use Metal GPU acceleration
                f16_kv=True,  # Use float16 for better performance
                use_mlock=True,  # Lock memory to prevent swapping
                n_threads=4,  # Optimize for Apple Silicon
            )
        elif device_type == "cpu":
            # CPU-only configuration
            llm = LlamaCpp(
                model_path=model_path,
                max_tokens=2048,
                n_ctx=4096,
                n_batch=512,
                callback_manager=None,
                verbose=False,
                n_gpu_layers=0,
                n_threads=8,  # Use more threads for CPU
            )
        else:
            # CUDA configuration
            llm = LlamaCpp(
                model_path=model_path,
                max_tokens=2048,
                n_ctx=4096,
                n_batch=512,
                callback_manager=None,
                verbose=False,
                n_gpu_layers=35,  # Use GPU layers for CUDA
            )
        
        LOGGING.info("GGUF/GGML model loaded successfully")
        return llm
        
    except Exception as e:
        LOGGING.error(f"Error loading GGUF/GGML model: {str(e)}")
        raise e


def load_quantized_model_awq(model_id, LOGGING):
    """
    Load an AWQ quantized model.
    Note: AWQ is primarily for NVIDIA GPUs, limited support on Mac.
    """
    try:
        LOGGING.info(f"Loading AWQ Model: {model_id}")
        
        # AWQ models are not well supported on Mac M4
        if platform.processor() == "arm":
            LOGGING.warning("AWQ models have limited support on Apple Silicon. Consider using GGUF models instead.")
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        
        LOGGING.info("AWQ model loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        LOGGING.error(f"Error loading AWQ model: {str(e)}")
        raise e


def load_quantized_model_qptq(model_id, model_basename, device_type, LOGGING):
    """
    Load a GPTQ quantized model.
    Note: GPTQ has limited support on Mac M4.
    """
    try:
        LOGGING.info(f"Loading GPTQ Model: {model_id}")
        
        if platform.processor() == "arm":
            LOGGING.warning("GPTQ models have limited support on Apple Silicon. Consider using GGUF models instead.")
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        
        if device_type == "mps":
            # MPS doesn't support GPTQ well, use CPU
            device_map = None
            torch_dtype = torch.float32
        else:
            device_map = "auto"
            torch_dtype = torch.float16
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        
        LOGGING.info("GPTQ model loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        LOGGING.error(f"Error loading GPTQ model: {str(e)}")
        raise e


def load_full_model(model_id, model_basename, device_type, LOGGING):
    """
    Load a full precision model, optimized for Mac M4.
    """
    try:
        LOGGING.info(f"Loading Full Model: {model_id}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        # Configure for Mac M4 optimization
        if device_type == "mps":
            # Optimize for Apple Silicon with Metal Performance Shaders
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,  # Use float16 for better performance
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            
            # Move model to MPS device
            if hasattr(model, 'to'):
                model = model.to('mps')
                
        elif device_type == "cpu":
            # CPU configuration
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                device_map=None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        else:
            # CUDA configuration
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        
        LOGGING.info("Full model loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        LOGGING.error(f"Error loading full model: {str(e)}")
        raise e


def get_model_recommendations(available_memory_gb: float) -> dict:
    """
    Get model recommendations based on available memory.
    
    Args:
        available_memory_gb: Available system memory in GB
        
    Returns:
        dict: Recommended models and configurations
    """
    recommendations = {
        "memory_gb": available_memory_gb,
        "models": []
    }
    
    if available_memory_gb >= 32:
        recommendations["models"].extend([
            {
                "name": "Llama-3-8B-Instruct",
                "id": "meta-llama/Meta-Llama-3-8B-Instruct",
                "size_gb": 16,
                "performance": "High",
                "recommended": True
            },
            {
                "name": "Mistral-7B-Instruct",
                "id": "mistralai/Mistral-7B-Instruct-v0.2",
                "size_gb": 14,
                "performance": "High", 
                "recommended": True
            }
        ])
    
    if available_memory_gb >= 16:
        recommendations["models"].extend([
            {
                "name": "Llama-3-8B-Instruct (GGUF Q4)",
                "basename": "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
                "size_gb": 4.5,
                "performance": "Good",
                "recommended": True
            }
        ])
    
    if available_memory_gb >= 8:
        recommendations["models"].extend([
            {
                "name": "Llama-3-8B-Instruct (GGUF Q2)",
                "basename": "Meta-Llama-3-8B-Instruct.Q2_K.gguf", 
                "size_gb": 3,
                "performance": "Moderate",
                "recommended": False
            }
        ])
    
    return recommendations


def download_model_if_needed(model_id: str, model_basename: str = None) -> str:
    """
    Download model if not present locally.
    
    Args:
        model_id: HuggingFace model ID
        model_basename: Model file basename for GGUF models
        
    Returns:
        str: Local path to model
    """
    models_dir = "./models"
    os.makedirs(models_dir, exist_ok=True)
    
    if model_basename:
        # GGUF model download
        model_path = os.path.join(models_dir, model_basename)
        if not os.path.exists(model_path):
            logging.info(f"Downloading GGUF model: {model_basename}")
            # Here you would implement download logic
            # For now, just log that manual download is needed
            logging.warning(f"Please manually download {model_basename} from {model_id} to {models_dir}/")
            
        return model_path
    else:
        # HuggingFace model (downloaded automatically by transformers)
        return model_id 