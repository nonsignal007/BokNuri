from langchain.docstore.document import Document
import re
from pypdf import PdfReader
from glob import glob

class LegalText:
    """법 전처리 클래스"""
    def __init__(self, pdf_path):
        self.docs = self.load_pdf_text(pdf_path)
        self.documents = []
        self.law_name = self.extract_law_title(self.docs)
        self.valid , self.valid_date = self.is_valid(self.docs)
        self.pre_doc = self.set_doc(self.docs)

    def load_pdf_text(self, pdf_path):
        reader = PdfReader(pdf_path)
        docs = ''
        law_name = pdf_path.split('/')[-1].split('.')[0]

        for page in reader.pages:
            temp = page.extract_text()
            temp = temp[temp.find('국가법령정보센터\n')+9:]
            temp = temp[temp.find('\n'):]
            docs += temp
        return docs

    def is_valid(self, text):
        """[시행 2024. 10. 22.]"""
        valid_pattern = r'\[시행\s(\d{4}\.\s\d{1,2}\.\s\d{1,2}\.)\]'
        match_pattern = re.search(valid_pattern, text)
        
        ## 날짜 형식으로 변경
        times = match_pattern.group(1).replace('.', '-').replace(' ', '')[:-1]
        if match_pattern:
            return True, times

    def extract_law_title(self, text):
        """제목 추출"""
        title_pattern = r'\n\s+\n(.+?)\n'
        title_match = re.search(title_pattern, text)
        if title_match:
            return title_match.group(1).strip()
        else:
            assert False, "제목을 찾을 수 없습니다."

    ##### 초기 설정 ######
    
    def extract_main_articles(self, text):
        """주요 조문 추출"""
        pattern = r'(제\d+조(?:의\d+)?\s*\([^)]+\))\s*(.*?)(?=\n\s*\n제\d+조(?:의\d+)?\s*\([^)]+\)|$)'
        match = re.findall(pattern, text, re.DOTALL)
        return match

    def set_doc(self, text):
        # 장(章) 제목 패턴
        pattern = r'제\d+장\s+([^\n]+)\n\s\n'
        chapter_list = re.split(pattern, text) ## 1장밖에 없음
        paragraph_title = []
        number = 1

        if len(chapter_list) == 1:
            for j in range(len(self.extract_main_articles(chapter_list[0]))):
                article = self.extract_main_articles(chapter_list[0])[j][0]
                article_number = article[:article.find('(')]
                article_subject = article[article.find('(')+1:article.find(')')]

                self.documents.append(Document(
                    page_content=self.extract_main_articles(chapter_list[0])[j][1],
                    metadata={
                        "law_title" : self.law_name, ## 법 제목
                        "effective_date" : self.valid_date, ## 시행 일자
                        "paragraph_number" : "", ## 장 번호
                        "paragraph_subject" : "", ## 항 제목
                        "article_number" : article_number, ## 조문 번호
                        "article_subject" : article_subject, ## 조문 주제
                        # "document_type" : "법률",
                        # "is_valid" : self.valid,
                        # "legal_area" : "장애인복지",
                    }
                ))
        
        else:
            for i in range(len(chapter_list)):
                ## 첫 장 패스
                if i == 0:
                    continue

                ## 장 제목 : 홀수
                if i % 2 != 0:
                    paragraph_title.append(f'제{number}장 {chapter_list[i]}')
                    number += 1
                    continue
                ## 조문 : 짝수
                else:
                    # print('=======')
                    # print(chapter_list[i])
                    # print(self.extract_main_articles(chapter_list[i]))

                    paragraph = paragraph_title.pop(0)
                    paragraph_number = paragraph.split()[0]
                    paragraph_subject = ' '.join(paragraph.split()[1])

                    for j in range(len(self.extract_main_articles(chapter_list[i]))):
                        article = self.extract_main_articles(chapter_list[i])[j][0]
                        article_number = article[:article.find('(')]
                        article_subject = article[article.find('(')+1:article.find(')')]

                        self.documents.append(Document(
                            page_content=self.extract_main_articles(chapter_list[i])[j][1],
                            metadata={
                                "law_title" : self.law_name, ## 법 제목
                                "effective_date" : self.valid_date, ## 시행 일자
                                "paragraph_number" : paragraph_number, ## 장 번호
                                "paragraph_subject" : paragraph_subject, ## 항 제목
                                "article_number" : article_number, ## 조문 번호
                                "article_subject" : article_subject, ## 조문 주제
                                # "document_type" : "법률",
                                # "is_valid" : self.valid,
                                # "legal_area" : "장애인복지",
                            }
                            )
                        )
            
        return self.documents
    
# filenames = glob('/Volumes/MINDB/24년/SW아카데미/LangEyE/App/src/crawling/files/laws/*.pdf')

# # print(filenames[0])
# # LegalText(filenames[0]).documents

# docs = []
# for filename in filenames:
#     docs.extend(LegalText(filename).documents)
# docs


from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Dict, Union
import re

class PDFDocumentProcessor:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        PDF 문서 처리를 위한 클래스 초기화
        
        Args:
            chunk_size (int): 텍스트 분할 크기
            chunk_overlap (int): 텍스트 분할 시 중복 크기
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def _extract_section_content(self, text: str, section: str) -> str:
        """
        텍스트에서 특정 섹션의 내용을 추출
        
        Args:
            text (str): 전체 텍스트
            section (str): 추출할 섹션 이름 (대상, 내용, 방법, 문의)
            
        Returns:
            str: 추출된 섹션 내용
        """
        pattern = f"{section}\\s*(.*?)(?=대상|내용|방법|문의|$)"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""
    
    def _process_page(self, page) -> Document:
        """
        PDF 페이지를 처리하여 Document 객체 생성
        
        Args:
            page: PDF 페이지 객체
            
        Returns:
            Document: 처리된 문서 객체
        """
        text = page.page_content
        
        # 섹션별 내용 추출
        sections = {
            "target": self._extract_section_content(text, "대상"),
            "content": self._extract_section_content(text, "내용"),
            "method": self._extract_section_content(text, "방법"),
            "contact": self._extract_section_content(text, "문의")
        }
        
        # 메타데이터 생성
        metadata = {
            "page": page.metadata["page"],
            **{f"section_{k}": v for k, v in sections.items()},
            **{f"has_{k}": bool(v) for k, v in sections.items()}
        }
        
        return Document(page_content=text, metadata=metadata)
    
    def process_pdf(self, pdf_path: str, split: bool = True) -> Union[List[Document], List[List[Document]]]:
        """
        PDF 파일을 처리하여 Document 리스트 반환
        
        Args:
            pdf_path (str): PDF 파일 경로
            split (bool): 청크 분할 여부
            
        Returns:
            Union[List[Document], List[List[Document]]]: 
            - split=True인 경우: 분할된 Document 리스트의 리스트 (페이지별로 그룹화)
            - split=False인 경우: Document 리스트 (페이지당 하나의 Document)
        """
        # PDF 로드
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        
        # 각 페이지 처리
        documents = [self._process_page(page) for page in pages]
        
        if not split:
            return documents
        
        # 페이지별로 문서 분할
        split_documents = []
        for doc in documents:
            splits = self.text_splitter.split_documents([doc])
            split_documents.append(splits)
        
        return split_documents

# # 사용 예시
# if __name__ == "__main__":
#     # 프로세서 인스턴스 생성
#     processor = PDFDocumentProcessor(
#         chunk_size=1000,
#         chunk_overlap=200
#     )
    
#     # 전체 문서 처리 (분할 없음)
#     full_documents = processor.process_pdf("your_pdf_path.pdf", split=False)
    
#     # 청크로 분할된 문서 처리
#     split_documents = processor.process_pdf("your_pdf_path.pdf", split=True)
    
#     # 처리된 문서 확인 예시
#     for doc in full_documents:
#         print(f"Page: {doc.metadata['page']}")
#         print(f"Has target section: {doc.metadata['has_target']}")
#         print(f"Target content: {doc.metadata['section_target']}")
#         print("-" * 50)