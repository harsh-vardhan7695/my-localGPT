from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate


def get_prompt_template(promptTemplate_type="llama3", history=False):
    """
    Get prompt template and memory configuration based on model type.
    
    Args:
        promptTemplate_type (str): Type of prompt template ("llama3", "llama", "mistral", "non_llama")
        history (bool): Whether to use conversation history
        
    Returns:
        tuple: (prompt_template, memory)
    """
    
    if promptTemplate_type == "llama3":
        B_INST, E_INST = "<|start_header_id|>user<|end_header_id|>", "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        B_SYS, E_SYS = "<|start_header_id|>system<|end_header_id|>", "<|eot_id|>"
        SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible using the context text provided. Your answers should only answer the question once and not have any text after the answer is done.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

        if history:
            instruction = """
Context: {history} \n {context}
User: {question}"""

            prompt_template = B_SYS + SYSTEM_PROMPT + E_SYS + B_INST + instruction + E_INST
        else:
            instruction = """
Context: {context}
User: {question}"""
            
            prompt_template = B_SYS + SYSTEM_PROMPT + E_SYS + B_INST + instruction + E_INST

    elif promptTemplate_type == "llama":
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible using the context text provided. Your answers should only answer the question once and not have any text after the answer is done.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

        if history:
            instruction = """
Context: {history} \n {context}
User: {question}"""

            prompt_template = B_INST + B_SYS + SYSTEM_PROMPT + E_SYS + instruction + E_INST
        else:
            instruction = """
Context: {context}
User: {question}"""
            
            prompt_template = B_INST + B_SYS + SYSTEM_PROMPT + E_SYS + instruction + E_INST

    elif promptTemplate_type == "mistral":
        B_INST, E_INST = "<s>[INST] ", " [/INST]"
        if history:
            prompt_template = (
                B_INST
                + "Always answer as helpfully as possible using the context text provided. "
                + "Your answers should only answer the question once and not have any text after the answer is done.\n\n"
                + "Context: {history} \n {context}\n"
                + "Question: {question}"
                + E_INST
            )
        else:
            prompt_template = (
                B_INST
                + "Always answer as helpfully as possible using the context text provided. "
                + "Your answers should only answer the question once and not have any text after the answer is done.\n\n"
                + "Context: {context}\n"
                + "Question: {question}"
                + E_INST
            )

    else:  # non_llama
        if history:
            prompt_template = """
Please use the following context to answer questions.
Context: {history} \n {context}
---
Question: {question}
Answer: """
        else:
            prompt_template = """
Please use the following context to answer questions.
Context: {context}
---
Question: {question}
Answer: """

    # Create memory if history is enabled
    memory = None
    if history:
        memory = ConversationBufferWindowMemory(
            input_key="question", 
            memory_key="history", 
            k=3
        )

    prompt = PromptTemplate(
        input_variables=["history", "context", "question"] if history else ["context", "question"],
        template=prompt_template,
    )

    return prompt, memory


def get_system_prompts():
    """
    Get various system prompts for different use cases.
    
    Returns:
        dict: Collection of system prompts
    """
    return {
        "general": """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible using the context text provided. Your answers should only answer the question once and not have any text after the answer is done.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
        
        "technical": """You are a technical documentation assistant. You provide accurate, detailed answers based on the provided context. Focus on technical accuracy and include relevant details. If information is missing from the context, clearly state that.""",
        
        "concise": """You are a concise assistant. Provide short, direct answers based solely on the provided context. Do not elaborate beyond what is explicitly stated in the context.""",
        
        "analytical": """You are an analytical assistant. Break down complex questions and provide structured answers based on the context. Organize your response logically and cite specific parts of the context when relevant."""
    }


def format_chat_prompt(question: str, context: str, model_type: str = "llama3") -> str:
    """
    Format a standalone chat prompt for direct model inference.
    
    Args:
        question (str): User's question
        context (str): Retrieved context
        model_type (str): Model type for formatting
        
    Returns:
        str: Formatted prompt
    """
    prompts = {
        "llama3": f"""<|start_header_id|>system<|end_header_id|>
You are a helpful assistant. Answer the question based on the provided context.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Context: {context}

Question: {question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",

        "mistral": f"""<s>[INST] Answer the question based on the provided context.

Context: {context}

Question: {question} [/INST]""",

        "llama": f"""[INST] <<SYS>>
Answer the question based on the provided context.
<</SYS>>

Context: {context}

Question: {question} [/INST]""",

        "general": f"""Based on the following context, please answer the question:

Context: {context}

Question: {question}

Answer:"""
    }
    
    return prompts.get(model_type, prompts["general"])


def get_evaluation_prompts():
    """
    Get prompts for evaluating response quality.
    
    Returns:
        dict: Evaluation prompts
    """
    return {
        "relevance": """Rate the relevance of this answer to the question on a scale of 1-5:
Question: {question}
Answer: {answer}
Context: {context}

Rating (1-5):""",

        "accuracy": """Evaluate if this answer is factually accurate based on the context:
Context: {context}
Answer: {answer}

Is the answer accurate? (Yes/No/Partially):""",

        "completeness": """Rate how complete this answer is on a scale of 1-5:
Question: {question}
Answer: {answer}

Completeness Rating (1-5):"""
    } 