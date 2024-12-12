import torch
from models.RAG.pdf_preprocessing import LegalText
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.embeddings.cache import CacheBackedEmbeddings
from glob import glob
import json
import os

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_pdf_documents(pdf_dir):
    """PDF 파일들을 읽어서 Document 객체 리스트로 변환합니다."""
    filenames = glob(f'{pdf_dir}/*.pdf')
    law_docs = []
    for filename in filenames:
        law_docs.extend(LegalText(filename).documents)
    return law_docs

def load_qna_data(file_path):
    """QnA JSON 파일을 읽어서 Document 객체 리스트로 변환합니다."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = []
    for item in data:
        metadata = {
            **item.get('metadata', {}),
            'question': item['question']
        }
        
        doc = Document(
            page_content=item['answer'],
            metadata=metadata
        )
        documents.append(doc)
    
    return documents

def create_cached_embeddings(base_embeddings):
    """CacheBackedEmbeddings 인스턴스를 생성합니다."""
    cache_dir = './weights/embedding_cache'
    os.makedirs(cache_dir, exist_ok=True)
    
    fs = LocalFileStore(cache_dir)
    
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=base_embeddings,
        document_store=fs,
        namespace="bge-m3-korean"
    )
    
    return cached_embeddings

def create_faiss_db(base_embedding_model):
    """Document 객체들로 새로운 FAISS DB를 생성합니다."""
    DB_PATH = "./faiss_db"
    PDF_DIR = "/workspace/LangEyE/App/src/crawling/files/laws"
    JSON_PATH = "/workspace/LangEyE/App/src/crawling/files/faq_results_20241211.json"

    # 캐시된 임베딩 모델 생성
    cached_embeddings = create_cached_embeddings(base_embedding_model)
    
    pdf_documents = load_pdf_documents(PDF_DIR)
    qna_documents = load_qna_data(JSON_PATH)
    
    # 모든 문서 합치기
    all_documents = pdf_documents + qna_documents
    
    # FAISS DB 생성
    db = FAISS.from_documents(
        documents=all_documents,
        embedding=cached_embeddings
    )
    
    # FAISS DB 저장
    os.makedirs(DB_PATH, exist_ok=True)
    db.save_local(DB_PATH)
    
    print(f"Successfully created FAISS DB with {len(all_documents)} total documents")
    return db

def load_faiss_db(base_embedding_model):
    """저장된 FAISS DB를 로드합니다."""
    DB_PATH = "./faiss_db"
    
    if not os.path.exists(DB_PATH):
        print("No existing FAISS DB found. Creating new one...")
        return create_faiss_db(base_embedding_model)
    
    # 캐시된 임베딩 모델 생성
    cached_embeddings = create_cached_embeddings(base_embedding_model)
    
    # 저장된 FAISS DB 로드
    db = FAISS.load_local(
        folder_path=DB_PATH,
        embeddings=cached_embeddings
    )
    
    print(f"Successfully loaded existing FAISS DB")
    return db