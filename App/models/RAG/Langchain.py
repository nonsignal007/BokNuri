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
from pdf_preprocessing import LegalText

# 캐시 디렉토리 설정
cache_dir = './weights'
os.makedirs(cache_dir, exist_ok=True)
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HOME'] = cache_dir
os.environ['HF_DATASETS_CACHE'] = os.path.join(cache_dir, 'datasets')
os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir
os.environ['TORCH_HOME'] = os.path.join(cache_dir, 'torch')

def load_rag_model():
    pdf_path = "/workspace/LangEyE/crawling/장애인복지법.pdf"
    # PDF 파일 로드
    docs = LegalText(pdf_path).documents

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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 프롬프트 템플릿 (사용자 정보 포함)
    template = """다음은 장애인복지법의 일부 내용입니다:
    {context}

    질문: {question}

    위 법령 내용을 바탕으로 답변해주세요. 
    - 관련 법조항을 명시해주세요
    - 법령에 명시된 내용만 답변해주세요
    - 불확실한 내용은 '법령에서 명확히 명시되어 있지 않습니다'라고 답변해주세요
    - 답변 마지막에 참고한 법조항 번호를 명시해주세요

    답변:"""
    
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
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain

    return create_chain_with_user_info


class ChatbotState:
    def __init__(self):
        """챗봇 상태 초기화"""
        self.create_chain = load_rag_model()
        self.chain = None
        self.settings = None
        self.initialize_chain()
    
    def initialize_chain(self):
        """기본 체인 초기화"""
        try:
            default_info = "설정되지 않은 사용자"
            self.chain = self.create_chain(default_info)
        except Exception as e:
            print(f"Chain initialization error: {str(e)}")
            self.chain = None

    def update_settings(self, settings):
        """사용자 설정 업데이트 및 체인 재생성"""
        try:
            self.settings = settings
            user_info = format_user_info(settings)
            self.chain = self.create_chain(user_info)
            return True
        except Exception as e:
            print(f"Settings update error: {str(e)}")
            return False

chatbot_state = ChatbotState()

text = "장애정도 재심사가 뭐야"

response = chatbot_state.chain.invoke(text)

print(response)