import torch
import bs4
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline

# URL 설정
from langchain.document_loaders import PyPDFLoader

# PDF 파일 로드. 파일의 경로 입력
loader = PyPDFLoader("guide_book.pdf")

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

# 단계 5: 모델 설정
model_id = "davidkim205/Ko-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

def generate_response(context, question):
    prompt = f"다음 컨텍스트를 바탕으로 질문에 답변해주세요:\n\n{context}\n\n질문: {question}"
    
    messages = [
        {"role": "system", "content": "당신은 구체적으로 답변하는 챗봇입니다."},
        {"role": "user", "content": prompt},
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=1024,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

# 단계 6: RAG 체인 실행
def run_rag_chain(question):
    # 문서 검색
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    # 응답 생성
    response = generate_response(context, question)
    
    return response

# 단계 7: 대화형 인터페이스
if __name__ == "__main__":
    print("대화를 시작합니다. 종료하려면 'quit' 또는 'exit'를 입력하세요.")
    while True:
        question = input('>')
        
        if question.lower() in ['quit', 'exit']:
            break
            
        response = run_rag_chain(question)
        print(response)
