import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

from langchain.llms import HuggingFacePipeline

def load_model(model_name, cache_dir):
    """로컬 LLM 모델 로드"""
    if model_name == 'llama':
        tokenizer = AutoTokenizer.from_pretrained(
            "davidkim205/Ko-Llama-3-8B-Instruct",
            cache_dir=cache_dir
        )
        model = AutoModelForCausalLM.from_pretrained(
            "davidkim205/Ko-Llama-3-8B-Instruct",
            device_map="auto",
            torch_dtype=torch.float16,
            cache_dir=cache_dir
        )
    elif model_name == 'qwen':
        tokenizer = AutoTokenizer.from_pretrained(
            "davidkim205/Ko-Qwen-3-8B-Instruct",
            cache_dir=cache_dir
        )
        model = AutoModelForCausalLM.from_pretrained(
            "davidkim205/Ko-Qwen-3-8B-Instruct",
            device_map="cuda",
            torch_dtype=torch.float16,
            cache_dir=cache_dir
        )
    
    # HuggingFace pipeline 생성
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0.1,
        top_p=0.95,
        repetition_penalty=1.15
    )
    
    # LangChain용 모델 래퍼 생성
    llm = HuggingFacePipeline(pipeline=pipe)

    return llm