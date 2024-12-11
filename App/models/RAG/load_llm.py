import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

from langchain_core.pipeline import HuggingFacePipeline

def load_model(self, model_name):
    """로컬 LLM 모델 로드"""
    if model_name == 'llama':
        self.tokenizer = AutoTokenizer.from_pretrained(
            "davidkim205/Ko-Llama-3-8B-Instruct",
            cache_dir=self.cache_dir
        )
        model = AutoModelForCausalLM.from_pretrained(
            "davidkim205/Ko-Llama-3-8B-Instruct",
            device_map="auto",
            torch_dtype=torch.float16,
            cache_dir=self.cache_dir
        )
    elif model_name == 'qwen':
        self.tokenizer = AutoTokenizer.from_pretrained(
            "davidkim205/Ko-Qwen-3-8B-Instruct",
            cache_dir=self.cache_dir
        )
        model = AutoModelForCausalLM.from_pretrained(
            "davidkim205/Ko-Qwen-3-8B-Instruct",
            device_map="cuda",
            torch_dtype=torch.float16,
            cache_dir=self.cache_dir
        )
    
    # HuggingFace pipeline 생성
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=self.tokenizer,
        max_length=2048,
        temperature=0.1,
        top_p=0.95,
        repetition_penalty=1.15
    )
    
    # LangChain용 모델 래퍼 생성
    llm = HuggingFacePipeline(pipeline=pipe)

    return llm