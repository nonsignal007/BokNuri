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
url = "https://n.news.naver.com/article/437/0000378416"

# 단계 1: 문서 로드
loader = WebBaseLoader(
    web_paths=(url,),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            "div",
            attrs={"class": ["newsct_article _article_body", "media_end_head_title"]},
        )
    ),
)
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
model_kwargs = {'device': device}
encode_kwargs = {'normalize_embeddings': True}

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
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
tokenizer = AutoTokenizer.from_pretrained("beomi/Llama-3-Open-Ko-8B")
model = AutoModelForCausalLM.from_pretrained(
    "beomi/Llama-3-Open-Ko-8B",
    device_map="auto",  # auto로 설정하여 accelerate가 알아서 처리하도록 함
    torch_dtype=torch.float16
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
question = "부영그룹의 출산 장려 정책에 대해 설명해주세요"
response = rag_chain.invoke(question)

# 결과 출력
print(f"URL: {url}")
print(f"검색된 문서 수: {len(docs)}")
print("===" * 20)
print(f"[HUMAN]\n{question}\n")
print(f"[AI]\n{response}")
