from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from glob import glob
import os
import json
import torch
from langchain.docstore.document import Document
from models.RAG.pdf_preprocessing import LegalText

def create_embedding_model():
    """임베딩 모델 생성"""
    # 캐시 디렉토리 설정
    cache_dir = './weights'
    os.makedirs(cache_dir, exist_ok=True)
    
    # 환경 변수 설정
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    os.environ['HF_HOME'] = cache_dir
    os.environ['HF_DATASETS_CACHE'] = os.path.join(cache_dir, 'datasets')
    os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir
    os.environ['TORCH_HOME'] = os.path.join(cache_dir, 'torch')

    # 디바이스 설정
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    # 임베딩 모델 생성
    return HuggingFaceEmbeddings(
        model_name="upskyy/bge-m3-korean",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True},
        cache_folder=cache_dir
    )

def create_law_db(embedding_model, save_path="/workspace/LangEyE/App/data/laws"):
    """법령 DB 생성"""
    try:
        documents = []
        filenames = glob('/workspace/LangEyE/App/src/crawling/files/laws/*.pdf')
        
        print(f"법령 PDF 파일 수: {len(filenames)}")
        
        for filename in filenames:
            try:
                legal_doc = LegalText(filename)
                documents.extend(legal_doc.documents)
                print(f"처리 완료: {filename}")
            except Exception as e:
                print(f"PDF 파일 처리 오류 ({filename}): {str(e)}")
                continue
        
        if not documents:
            raise ValueError("처리된 문서가 없습니다")
            
        print(f"총 처리된 법령 문서 수: {len(documents)}")
        db = FAISS.from_documents(documents, embedding_model)
        os.makedirs(save_path, exist_ok=True)
        db.save_local(save_path)
        print(f"법령 DB 저장 완료: {save_path}")
        return db
        
    except Exception as e:
        print(f"법령 DB 생성 오류: {str(e)}")
        return None

def create_qa_db(embedding_model, save_path="/workspace/LangEyE/App/data/qa"):
    """QA DB 생성"""
    try:
        documents = []
        qa_files = glob('/workspace/LangEyE/App/src/crawling/files/faq_results_20241211.json')
        
        print(f"QA JSON 파일 수: {len(qa_files)}")
        
        for qa_file in qa_files:
            try:
                with open(qa_file, 'r', encoding='utf-8') as f:
                    qa_data = json.load(f)
                
                for qa in qa_data:
                    doc = Document(
                        page_content=qa.get('answer', ''),
                        metadata={
                            'doc_type': 'qa',
                            'question': qa.get('question', ''),
                            'category': qa.get('category', 'general')
                        }
                    )
                    documents.append(doc)
                print(f"처리 완료: {qa_file}")
            except Exception as e:
                print(f"QA 파일 처리 오류 ({qa_file}): {str(e)}")
                continue
        
        if not documents:
            raise ValueError("처리된 문서가 없습니다")
            
        print(f"총 처리된 QA 문서 수: {len(documents)}")
        db = FAISS.from_documents(documents, embedding_model)
        os.makedirs(save_path, exist_ok=True)
        db.save_local(save_path)
        print(f"QA DB 저장 완료: {save_path}")
        return db
        
    except Exception as e:
        print(f"QA DB 생성 오류: {str(e)}")
        return None

def create_dbs():
    """법령과 QA DB 생성"""
    try:
        print("임베딩 모델 로딩 중...")
        embedding_model = create_embedding_model()
        
        print("\n=== 법령 DB 생성 시작 ===")
        law_db = create_law_db(embedding_model)
        
        print("\n=== QA DB 생성 시작 ===")
        qa_db = create_qa_db(embedding_model)
        
        return {
            'laws': law_db,
            'qa': qa_db
        }
        
    except Exception as e:
        print(f"DB 생성 중 오류 발생: {str(e)}")
        return None

if __name__ == "__main__":
    dbs = create_dbs()
    if dbs:
        print("\n법령과 QA DB 생성 완료!")
    else:
        print("\nDB 생성 실패!")