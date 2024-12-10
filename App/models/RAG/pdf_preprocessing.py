from langchain.docstore.document import Document
import re
from pypdf import PdfReader

class LegalText:
    def __init__(self, pdf_path):
        self.docs = self.load_pdf_text(pdf_path)
        self.documents = []
        self.law_name = self.extract_law_title(self.docs)
        self.valid , self.valid_date = self.is_valid(self.docs)
        self.pre_doc = self.set_doc(self.docs)

    def load_pdf_text(self, pdf_path):
        reader = PdfReader(pdf_path)
        docs = ''
        for page in reader.pages:
            temp = page.extract_text()
            temp = temp[temp.find('장애인복지법')+6:]
            docs += temp
        return docs

    def is_valid(self, text):
        """[시행 2024. 10. 22.]"""
        valid_pattern = r'\[시행\s(\d{4}\.\s\d{2}\.\s\d{2}\.)\]'
        match_pattern = re.search(valid_pattern, text)
        
        ## 날짜 형식으로 변경
        times = match_pattern.group(1).replace('.', '-').replace(' ', '')[:-1]

        if match_pattern:
            return True, times

    def extract_law_title(self, text):
        """제목 추출"""
        title_pattern = r'\n(.+?)\n'
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
        # print(match[0])
        return match

    def set_doc(self, text):
        # 장(章) 제목 패턴
        pattern = r'제\d+장\s+([^\n]+)\n\s\n'
        chapter_list = re.split(pattern, text)
        paragraph_title = []
        number = 1
        
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

                for j in range(len(self.extract_main_articles(chapter_list[i]))):
                    self.documents.append(Document(
                        page_content=self.extract_main_articles(chapter_list[i])[j][1],
                        metadata={
                            "law_title" : self.law_name, ## 법 제목
                            "effective_date" : self.valid_date, ## 시행 일자
                            "paragraph" : paragraph, ## 항 제목
                            "article_number" : self.extract_main_articles(chapter_list[i])[j][0], ## 조문 번호
                            "document_type" : "법률",
                            "is_valid" : self.valid,
                            "legal_area" : "장애인복지",
                        }
                    ))

        return self.documents