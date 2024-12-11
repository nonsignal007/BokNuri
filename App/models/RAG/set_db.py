import torch
from load_embedding import load_embedding
from pdf_preprocessing import LegalText
from langchain_core.documents import Document
from langchain_chroma import Chroma
from glob import glob
import json

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

def create_chroma_db(embedding_model):
    """Document 객체들로 새로운 Chroma DB를 생성합니다."""
    DB_PATH = "./chroma_db"
    PDF_DIR = "/workspace/LangEyE/App/src/crawling/files/laws"
    JSON_PATH = "/workspace/LangEyE/App/src/crawling/files/faq_results_20241211.json"

    device = get_device()
    
    pdf_documents = load_pdf_documents(PDF_DIR)
    qna_documents = load_qna_data(JSON_PATH)
    
    # 모든 문서 합치기
    all_documents = pdf_documents + qna_documents
    
    db = Chroma.from_documents(
        documents=all_documents,
        embedding=embedding_model,
        persist_directory=DB_PATH,
        collection_name="my_db"
    )
    db.persist()
    print(f"Successfully created Chroma DB with {len(all_documents)} total documents")
    return db