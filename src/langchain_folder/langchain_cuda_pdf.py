import torch
import bs4
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# URL 설정
from langchain.document_loaders import PyPDFLoader

# cache 설정
import os

# 모든 가중치 파일의 캐시 디렉토리 설정
cache_dir = './weights'
os.makedirs(cache_dir, exist_ok=True)

# HuggingFace 관련 모든 캐시 경로 설정
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HOME'] = cache_dir
os.environ['HF_DATASETS_CACHE'] = os.path.join(cache_dir, 'datasets')
os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir
os.environ['TORCH_HOME'] = os.path.join(cache_dir, 'torch')

# 설정 확인
print("TRANSFORMERS_CACHE:", os.getenv('TRANSFORMERS_CACHE'))
print("HF_HOME:", os.getenv('HF_HOME'))
print("HF_DATASETS_CACHE:", os.getenv('HF_DATASETS_CACHE'))
print("HUGGINGFACE_HUB_CACHE:", os.getenv('HUGGINGFACE_HUB_CACHE'))
print("TORCH_HOME:", os.getenv('TORCH_HOME'))

# PDF 파일 로드. 파일의 경로 입력
loader = PyPDFLoader("/workspace/LangEyE/langchain_folder/pdf/장애인ㆍ노인ㆍ임산부 등의 편의증진 보장에 관한 법률 시행규칙.pdf")

# 페이지 별 문서 로드
docs = loader.load()

# 단계 2: 문서 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
splits = text_splitter.split_documents(docs)

# 단계 3: 임베딩 설정
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

model_name = "upskyy/bge-m3-korean"
model_kwargs = {'device': device }
encode_kwargs = {'normalize_embeddings': True}

# 임베딩 모델
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    cache_folder=cache_dir  # 캐시 디렉토리 지정
)

vectorstore = FAISS.from_documents(
    documents=splits,
    embedding=embeddings
)




# 단계 4: 검색 설정
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)

# 단계 5: 프롬프트 템플릿
prompt = PromptTemplate.from_template("""
질문: {question}

답변:
{context}


""")

# 단계 6: 모델 설정
# tokenizer = AutoTokenizer.from_pretrained("beomi/Llama-3-Open-Ko-8B")
# model = AutoModelForCausalLM.from_pretrained(
#     "beomi/Llama-3-Open-Ko-8B",
#     device_map="auto",  # auto로 설정하여 accelerate가 알아서 처리하도록 함
#     torch_dtype=torch.float16
# )

# tokenizer와 모델
tokenizer = AutoTokenizer.from_pretrained(
    "davidkim205/Ko-Llama-3-8B-Instruct",
    cache_dir=cache_dir  # 캐시 디렉토리 지정
)

model = AutoModelForCausalLM.from_pretrained(
    "davidkim205/Ko-Llama-3-8B-Instruct",
    device_map="auto",
    torch_dtype=torch.float16,
    cache_dir=cache_dir  # 캐시 디렉토리 지정
)

# 파이프라인 설정 - device 인자 제거
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

# LangChain 모델 래퍼
llm = HuggingFacePipeline(pipeline=pipe)

# 단계 7: RAG 체인 생성
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 단계 8: 실행
question = "발달장애인 부모 상담 지원을 어디서 받을 수 있을까?"
response = rag_chain.invoke(question)

# 결과 출력

print(f"검색된 문서 수: {len(docs)}")
print("===" * 20)
print(f"[HUMAN]\n{question}\n")
print(f"[AI]\n{response}")

