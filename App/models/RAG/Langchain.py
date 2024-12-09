import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.text_splitter import SemanticChunker

# 캐시 디렉토리 설정
cache_dir = './weights'
os.makedirs(cache_dir, exist_ok=True)
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HOME'] = cache_dir
os.environ['HF_DATASETS_CACHE'] = os.path.join(cache_dir, 'datasets')
os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir
os.environ['TORCH_HOME'] = os.path.join(cache_dir, 'torch')

def load_rag_model():
    pdf_path = "/workspace/LangEyE/crawling/2024_장애인_안내책자.pdf"
    # PDF 파일 로드
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # 임베딩 모델 설정
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    model_name = "upskyy/bge-m3-korean"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True},
        cache_folder=cache_dir
    )

    # 임베딩 기반 문서 분할
    text_splitter = SemanticChunker(embeddings)
    splits = text_splitter.split_documents(docs)
    
    # 벡터스토어
    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 프롬프트 템플릿 (사용자 정보 포함)
    template = """
    당신은 시각장애인을 대상으로 복지 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다.

    사용자 정보:
    {user_info}

    질문: {question}

    관련 문서:
    {context}

    위 정보를 바탕으로 사용자의 상황에 맞는 복지 정보를 [관련 법안 혹은 근거], [대상], [혜택 내용], [신청 방법], [문의처]를 포함하여 안내해주세요.
    특히 사용자의 장애등급, 소득분위 등을 고려하여 맞춤형 정보를 제공해주세요.
    """
    
    prompt = PromptTemplate.from_template(template)

    # 모델 로드
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
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # RAG 체인 생성 (사용자 정보 포함)
    def create_chain_with_user_info(user_info: str):
        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
                "user_info": lambda _: user_info
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain

    return create_chain_with_user_info