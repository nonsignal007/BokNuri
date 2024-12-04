from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_pdf(path):    
    # PDF 파일 로드. 파일의 경로 입력
    loader = PyPDFLoader(path)

    # 페이지 별 문서 로드
    docs = loader.load()

    print(f"""
          문서의 수 : {len(docs)}
          PDF 파일을 로드하였습니다.
          """)

    return docs

def doc_split(docs, spliter, chunk_size=500, chunk_overlap=20):
    # 단계 2: 문서 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(docs)

    print(f"""
          문서의 수 : {len(splits)}
          문서를 분할하였습니다.
          """)

    return splits