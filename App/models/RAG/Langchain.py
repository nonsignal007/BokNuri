import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
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
    template = """장애인복지법 관련 질의응답을 진행하겠습니다.
    
    참고할 법령 내용:
    {context}
    
    질문: {question}
    
    답변 규칙:
    1. 위 법령 내용만을 기반으로 답변합니다.
    2. 답변의 근거가 되는 조항(예: 제00조 제0항)을 반드시 먼저 명시합니다.
    3. 조항의 내용을 직접 인용하면서 설명합니다.
    4. 법령에 명시되지 않은 내용은 추정하거나 해석하지 않고, "해당 내용은 제시된 법령에 명시되어 있지 않습니다"라고 답변합니다.
    
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