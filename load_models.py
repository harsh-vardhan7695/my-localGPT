import logging
import os
import torch
from typing import Optional, Tuple, Any

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    pipeline
)

# Try to import BitsAndBytesConfig for quantization
try:
    from transformers import BitsAndBytesConfig
    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False
    logging.warning("BitsAndBytesConfig not available. Quantization will be disabled.")

from huggingface_hub import hf_hub_download, snapshot_download
from langchain.llms import HuggingFacePipeline
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from constants import MODELS_PATH, MODEL_ID, MODEL_BASENAME, N_GPU_LAYERS, N_BATCH, CONTEXT_WINDOW_SIZE, MAX_NEW_TOKENS

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logging.warning("llama-cpp-python not available. GGUF models will not be supported.")

def detect_device():
    """
    Detect the best available device for model inference.
    Returns device type and whether it supports GPU acceleration.
    """
    if torch.cuda.is_available():
        return "cuda", True
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps", True
    else:
        return "cpu", False

def load_model_from_hf(
    model_id: str,
    device_type: str = "auto",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    trust_remote_code: bool = True,
    torch_dtype: str = "auto",
    low_cpu_mem_usage: bool = True,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
    logging_obj: Any = logging
) -> Tuple[Any, Any]:
    """
    Load a model and tokenizer from HuggingFace Hub with automatic download.
    
    Args:
        model_id: HuggingFace model identifier (e.g., "microsoft/DialoGPT-medium")
        device_type: Target device ("auto", "cuda", "mps", "cpu")
        load_in_8bit: Whether to load model in 8-bit precision
        load_in_4bit: Whether to load model in 4-bit precision
        trust_remote_code: Whether to trust remote code
        torch_dtype: PyTorch data type ("auto", "float16", "float32", "bfloat16")
        low_cpu_mem_usage: Use low CPU memory usage loading
        cache_dir: Directory to cache downloaded models
        local_files_only: Only use local files, don't download
        logging_obj: Logging object for messages
    
    Returns:
        Tuple of (model, tokenizer)
    """
    
    # Auto-detect device if needed
    if device_type == "auto":
        device_type, _ = detect_device()
    
    logging_obj.info(f"Loading model: {model_id}")
    logging_obj.info(f"Target device: {device_type}")
    
    # Configure quantization if requested
    quantization_config = None
    if load_in_8bit and load_in_4bit:
        raise ValueError("Cannot use both 8-bit and 4-bit quantization")
    
    if load_in_8bit or load_in_4bit:
        if QUANTIZATION_AVAILABLE:
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=load_in_8bit,
                    load_in_4bit=load_in_4bit,
                    bnb_4bit_compute_dtype=torch.float16 if load_in_4bit else None,
                    bnb_4bit_use_double_quant=True if load_in_4bit else None,
                    bnb_4bit_quant_type="nf4" if load_in_4bit else None,
                )
            except Exception as e:
                logging_obj.warning(f"Quantization configuration failed: {e}")
                quantization_config = None
        else:
            logging_obj.warning("Quantization requested but BitsAndBytesConfig not available")
            quantization_config = None
    
    # Configure torch dtype
    if torch_dtype == "auto":
        if device_type == "cuda":
            torch_dtype = torch.float16
        elif device_type == "mps":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
    elif isinstance(torch_dtype, str):
        torch_dtype = getattr(torch, torch_dtype, torch.float32)
    
    try:
        # Download and cache the model
        if not local_files_only:
            logging_obj.info(f"Downloading model files for {model_id}...")
            snapshot_download(
                repo_id=model_id,
                cache_dir=cache_dir,
                local_files_only=False,
                ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.safetensors"]
            )
        
        # Load tokenizer
        logging_obj.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            padding_side="left"
        )
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        logging_obj.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            device_map="auto" if device_type != "cpu" else None,
            low_cpu_mem_usage=low_cpu_mem_usage,
            cache_dir=cache_dir,
            local_files_only=local_files_only
        )
        
        # Move to device if not using device_map
        if device_type != "cpu" and quantization_config is None:
            model = model.to(device_type)
        
        logging_obj.info(f"Model loaded successfully on {device_type}")
        return model, tokenizer
        
    except Exception as e:
        logging_obj.error(f"Error loading model from HuggingFace: {e}")
        raise

def load_gguf_model(
    model_id: str,
    model_basename: str,
    device_type: str = "auto",
    n_ctx: int = 4096,
    n_batch: int = 512,
    n_gpu_layers: int = 0,
    verbose: bool = False,
    cache_dir: Optional[str] = None,
    logging_obj: Any = logging
) -> Any:
    """
    Load a GGUF model using llama-cpp-python with automatic download.
    
    Args:
        model_id: HuggingFace model identifier for GGUF model
        model_basename: Specific GGUF file name to download
        device_type: Target device type
        n_ctx: Context window size
        n_batch: Batch size
        n_gpu_layers: Number of layers to offload to GPU
        verbose: Whether to enable verbose logging
        cache_dir: Directory to cache downloaded models
        logging_obj: Logging object for messages
    
    Returns:
        Llama model instance
    """
    
    if not LLAMA_CPP_AVAILABLE:
        raise ImportError("llama-cpp-python is required for GGUF models")
    
    # Auto-detect device if needed
    if device_type == "auto":
        device_type, gpu_available = detect_device()
        if not gpu_available:
            n_gpu_layers = 0
    
    logging_obj.info(f"Loading GGUF model: {model_id}")
    logging_obj.info(f"Model file: {model_basename}")
    
    try:
        # Download the specific GGUF file
        logging_obj.info("Downloading GGUF model file...")
        model_path = hf_hub_download(
            repo_id=model_id,
            filename=model_basename,
            cache_dir=cache_dir,
            local_files_only=False
        )
        
        # Configure device-specific parameters
        if device_type == "cuda":
            n_gpu_layers = n_gpu_layers if n_gpu_layers > 0 else 35
        elif device_type == "mps":
            n_gpu_layers = n_gpu_layers if n_gpu_layers > 0 else 1
        else:
            n_gpu_layers = 0
        
        # Load the model
        logging_obj.info(f"Loading GGUF model from: {model_path}")
        llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
            n_threads=os.cpu_count() or 4,
            use_mmap=True,
            use_mlock=False,
        )
        
        logging_obj.info(f"GGUF model loaded successfully with {n_gpu_layers} GPU layers")
        return llm
        
    except Exception as e:
        logging_obj.error(f"Error loading GGUF model: {e}")
        raise

def create_hf_pipeline(
    model: Any,
    tokenizer: Any,
    device_type: str = "auto",
    max_new_tokens: int = 2048,
    temperature: float = 0.1,
    top_p: float = 0.95,
    top_k: int = 40,
    repetition_penalty: float = 1.15,
    do_sample: bool = True,
    logging_obj: Any = logging
) -> HuggingFacePipeline:
    """
    Create a HuggingFace pipeline for text generation.
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        device_type: Target device type
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Repetition penalty
        do_sample: Whether to use sampling
        logging_obj: Logging object for messages
    
    Returns:
        HuggingFacePipeline instance
    """
    
    # Auto-detect device if needed
    if device_type == "auto":
        device_type, _ = detect_device()
    
    try:
        # Create generation config
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        # Create pipeline (don't specify device when using accelerate)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            model_kwargs={"torch_dtype": model.dtype if hasattr(model, 'dtype') else torch.float32}
        )
        
        # Create callback manager
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        
        # Create LangChain pipeline
        hf_pipeline = HuggingFacePipeline(
            pipeline=pipe,
            callbacks=callback_manager,
            model_kwargs={"temperature": temperature, "max_new_tokens": max_new_tokens}
        )
        
        logging_obj.info("HuggingFace pipeline created successfully")
        return hf_pipeline
        
    except Exception as e:
        logging_obj.error(f"Error creating HuggingFace pipeline: {e}")
        raise

def load_model_auto(
    model_id: str,
    model_basename: Optional[str] = None,
    device_type: str = "auto",
    quantization: str = "none",
    max_new_tokens: int = 2048,
    cache_dir: Optional[str] = None,
    logging_obj: Any = logging
) -> HuggingFacePipeline:
    """
    Automatically load a model with the best configuration for the given device.
    
    Args:
        model_id: HuggingFace model identifier
        model_basename: Optional GGUF filename for quantized models
        device_type: Target device type ("auto", "cuda", "mps", "cpu")
        quantization: Quantization type ("none", "8bit", "4bit", "gguf")
        max_new_tokens: Maximum tokens to generate
        cache_dir: Cache directory for models
        logging_obj: Logging object for messages
    
    Returns:
        HuggingFacePipeline ready for use
    """
    
    # Auto-detect device if needed
    if device_type == "auto":
        device_type, gpu_available = detect_device()
        logging_obj.info(f"Auto-detected device: {device_type} (GPU: {gpu_available})")
    
    # Handle GGUF models
    if model_basename and (".gguf" in model_basename.lower() or ".ggml" in model_basename.lower()):
        if quantization == "none":
            quantization = "gguf"
    
    try:
        if quantization == "gguf":
            if not model_basename:
                raise ValueError("model_basename required for GGUF models")
            
            # Load GGUF model
            llm = load_gguf_model(
                model_id=model_id,
                model_basename=model_basename,
                device_type=device_type,
                cache_dir=cache_dir,
                logging_obj=logging_obj
            )
            
            # Create a simple wrapper for GGUF models
            class GGUFPipeline:
                def __init__(self, llm):
                    self.llm = llm
                
                def __call__(self, prompt, **kwargs):
                    max_tokens = kwargs.get('max_new_tokens', max_new_tokens)
                    temperature = kwargs.get('temperature', 0.1)
                    
                    response = self.llm(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stop=["</s>", "<|endoftext|>", "\n\n"],
                        echo=False
                    )
                    return response['choices'][0]['text']
            
            return GGUFPipeline(llm)
            
        else:
            # Load HuggingFace model
            load_in_8bit = quantization == "8bit"
            load_in_4bit = quantization == "4bit"
            
            model, tokenizer = load_model_from_hf(
                model_id=model_id,
                device_type=device_type,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                cache_dir=cache_dir,
                logging_obj=logging_obj
            )
            
            # Create HuggingFace pipeline
            hf_pipeline = create_hf_pipeline(
                model=model,
                tokenizer=tokenizer,
                device_type=device_type,
                max_new_tokens=max_new_tokens,
                logging_obj=logging_obj
            )
            
            return hf_pipeline
            
    except Exception as e:
        logging_obj.error(f"Error in load_model_auto: {e}")
        raise

# Legacy function wrappers for backward compatibility
def load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING):
    """Legacy wrapper for GGUF/GGML model loading"""
    return load_gguf_model(
        model_id=model_id,
        model_basename=model_basename,
        device_type=device_type,
        logging_obj=LOGGING
    )

def load_quantized_model_awq(model_id, LOGGING):
    """Legacy wrapper for AWQ model loading"""
    model, tokenizer = load_model_from_hf(
        model_id=model_id,
        load_in_4bit=True,
        logging_obj=LOGGING
    )
    return model, tokenizer

def load_quantized_model_qptq(model_id, model_basename, device_type, LOGGING):
    """Legacy wrapper for GPTQ model loading"""
    model, tokenizer = load_model_from_hf(
        model_id=model_id,
        load_in_4bit=True,
        device_type=device_type,
        logging_obj=LOGGING
    )
    return model, tokenizer

def load_full_model(model_id, model_basename, device_type, LOGGING):
    """Legacy wrapper for full model loading"""
    model, tokenizer = load_model_from_hf(
        model_id=model_id,
        device_type=device_type,
        logging_obj=LOGGING
    )
    return model, tokenizer

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