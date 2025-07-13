import logging
import os
import torch
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from constants import EMBEDDING_MODEL_NAME
from langchain.embeddings import HuggingFaceBgeEmbeddings


def get_embeddings(device_type: str = "mps"):
    """
    Get embeddings for the specified device type, optimized for Mac M4 with Metal Performance Shaders.
    
    Args:
        device_type (str): Device type - 'mps' for Mac M4, 'cpu', or 'cuda'
    
    Returns:
        Embeddings object
    """
    # Check if Metal Performance Shaders is available for Mac M4
    if device_type == "mps" and not torch.backends.mps.is_available():
        logging.warning("MPS not available, falling back to CPU")
        device_type = "cpu"
    
    # Determine device for embeddings
    if device_type == "cpu":
        device = "cpu"
    elif device_type == "mps":
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logging.info(f"Using device: {device} for embeddings")
    
    # Configure model kwargs for the device
    model_kwargs = {"device": device}
    # Note: torch_dtype is not supported in older versions of sentence-transformers
    # if device == "mps":
    #     # Optimize for Mac M4 Metal Performance Shaders
    #     model_kwargs.update({
    #         "torch_dtype": torch.float16,  # Use float16 for better performance on Apple Silicon
    #     })
    
    encode_kwargs = {"normalize_embeddings": True}
    
    # Choose embedding model based on EMBEDDING_MODEL_NAME
    if "instructor" in EMBEDDING_MODEL_NAME.lower():
        # For instructor models
        embeddings = HuggingFaceInstructEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            query_instruction="Represent this query for retrieval: ",
            embed_instruction="Represent this document for retrieval: "
        )
    else:
        # For other HuggingFace models
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    
    return embeddings

def check_system_compatibility():
    """
    Check system compatibility for Mac M4 LocalGPT setup.
    
    Returns:
        dict: System information and compatibility status
    """
    system_info = {
        "python_version": torch.__version__,
        "torch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    # Check if running on Apple Silicon
    import platform
    system_info["platform"] = platform.platform()
    system_info["processor"] = platform.processor()
    system_info["is_apple_silicon"] = platform.processor() == "arm"
    
    # Memory information
    import psutil
    system_info["total_memory_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
    system_info["available_memory_gb"] = round(psutil.virtual_memory().available / (1024**3), 2)
    
    return system_info

def log_system_info():
    """Log system information for debugging."""
    info = check_system_compatibility()
    logging.info("=== System Information ===")
    for key, value in info.items():
        logging.info(f"{key}: {value}")
    logging.info("========================")
    
    return info

def get_optimal_device():
    """
    Get the optimal device for the current system.
    
    Returns:
        str: Optimal device type ('mps', 'cuda', or 'cpu')
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def estimate_memory_usage(model_size_gb: float, embedding_size_gb: float = 1.5) -> dict:
    """
    Estimate memory usage for LocalGPT components.
    
    Args:
        model_size_gb: Size of the LLM model in GB
        embedding_size_gb: Size of embedding model in GB
        
    Returns:
        dict: Memory usage estimates
    """
    import psutil
    
    total_memory = psutil.virtual_memory().total / (1024**3)
    
    # Estimate memory usage
    estimated_usage = {
        "llm_model_gb": model_size_gb,
        "embedding_model_gb": embedding_size_gb,
        "vector_db_gb": 0.5,  # Estimated ChromaDB overhead
        "system_overhead_gb": 2.0,  # OS and other processes
        "total_estimated_gb": model_size_gb + embedding_size_gb + 0.5 + 2.0,
        "available_memory_gb": total_memory,
        "memory_sufficient": (model_size_gb + embedding_size_gb + 2.5) < (total_memory * 0.8)
    }
    
    return estimated_usage 