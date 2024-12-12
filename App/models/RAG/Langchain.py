from pathlib import Path
import yaml
from typing import Dict, List, Any, Optional
import logging

class DisabilityPromptManager:
   def __init__(self):
       self.base_path = '/workspace/LangEyE/App/models/RAG/prompt'
       self.templates = {
           'registration': f'{self.base_path}/disability_registration.yaml',
           'parking': f'{self.base_path}/disability_parking.yaml',
           'pension': f'{self.base_path}/disability_pension.yaml',
           'activity': f'{self.base_path}/disability_activity.yaml',
           'child': f'{self.base_path}/disability_child.yaml',
           'welfare': f'{self.base_path}/disability_welfare.yaml'
       }
       
       self.keywords = {
           'registration': [
               '등록', '신청', '장애인 등록', '장애등록', '등록절차', 
               '장애진단', '진단서', '장애심사', '등급심사'
           ],
           'parking': [
               '주차', '표지', '주차표지', '장애인주차', '편의', '편의시설',
               '주차구역', '주차카드', '장애인전용주차'
           ],
           'pension': [
               '연금', '수당', '장애연금', '장애수당', '급여', '기초연금',
               '복지급여', '연금신청', '수당신청'
           ],
           'activity': [
               '활동', '활동지원', '활동보조', '활동보조인', '활동지원사',
               '돌봄', '요양', '활동보조서비스'
           ],
           'child': [
               '아동', '장애아', '장애아동', '발달', '양육', '보육',
               '특수교육', '장애아가족', '장애아동재활'
           ],
           'welfare': [
               '복지', '복지사업', '지원사업', '바우처', '복지카드',
               '보조기구', '보장구', '일자리', '취업'
           ]
       }

   def classify_question(self, question: str) -> str:
       scores = {category: 0 for category in self.keywords.keys()}
       
       for category, keywords in self.keywords.items():
           for keyword in keywords:
               if keyword in question:
                   scores[category] += 1
                   if f"장애인 {keyword}" in question or f"장애 {keyword}" in question:
                       scores[category] += 2
       
       return max(scores.items(), key=lambda x: x[1])[0] if max(scores.values()) > 0 else 'welfare'

   def get_template_path(self, question: str) -> str:
       category = self.classify_question(question)
       return self.templates[category]

class LegalQASystem:
   def __init__(
       self,
       custom_llm,
       custom_embeddings,
       custom_db,
       template_path: str = "templates/qa_prompt.yaml"
   ):
       self.llm = custom_llm
       self.embeddings = custom_embeddings
       self.db = custom_db
       self.default_template_path = template_path
       self.prompt_manager = DisabilityPromptManager()
       self.prompt_template = None

   def _load_template(self, question: str) -> dict:
       """질문에 따라 적절한 템플릿을 동적으로 로드합니다."""
       try:
           template_path = self.prompt_manager.get_template_path(question)
           logging.info(f"Selected template: {template_path}")
           
           with open(template_path, 'r', encoding='utf-8') as f:
               template_data = yaml.safe_load(f)
           return template_data
       except FileNotFoundError:
           with open(self.default_template_path, 'r', encoding='utf-8') as f:
               return yaml.safe_load(f)
       except Exception as e:
           raise ValueError(f"템플릿 로드 중 오류 발생: {str(e)}")

   def _format_prompt(self, context: str, question: str) -> str:
       """템플릿을 사용하여 프롬프트를 포맷팅합니다."""
       try:
           if not self.prompt_template:
               self.prompt_template = self._load_template(question)
           
           template = self.prompt_template['template']
           return template.format(
               context=context,
               question=question
           )
       except KeyError:
           raise KeyError("템플릿에 'template' 키가 없습니다.")
       except Exception as e:
           raise ValueError(f"프롬프트 포맷팅 중 오류 발생: {str(e)}")

   def _search_relevant_laws(self, question: str) -> List[Dict[str, Any]]:
       """메타데이터와 컨텐츠를 결합한 검색을 수행합니다."""
       try:
           category = self.prompt_manager.classify_question(question)
           category_filters = {
               'registration': ["등록", "신청", "절차"],
               'parking': ["주차", "표지", "편의시설"],
               'pension': ["연금", "수당", "급여"],
               'activity': ["활동", "지원", "보조"],
               'child': ["아동", "양육", "보육"],
               'welfare': ["복지", "지원", "서비스"]
           }

           metadata_filters = {
               "article_subject": category_filters.get(category, ["신청", "절차", "방법"])
           }

           # 1. 메타데이터 기반 검색
           metadata_results = self.db.similarity_search(
               query=question,
               k=5,
               filter={"article_subject": {"$in": metadata_filters["article_subject"]}}
           )
           
           # 2. 일반 컨텐츠 기반 검색
           content_results = self.db.similarity_search(
               query=question,
               k=5
           )
           
           # 3. 결과 병합 및 정렬
           all_results = []
           seen_contents = set()
           
           # 메타데이터 결과 처리 (높은 우선순위)
           for doc in metadata_results:
               if doc.page_content not in seen_contents:
                   seen_contents.add(doc.page_content)
                   all_results.append({
                       'content': doc.page_content,
                       'metadata': {
                           'law_title': doc.metadata.get('law_title', ''),
                           'effective_date': doc.metadata.get('effective_date', ''),
                           'paragraph_number': doc.metadata.get('paragraph_number', ''),
                           'paragraph_subject': doc.metadata.get('paragraph_subject', ''),
                           'article_number': doc.metadata.get('article_number', ''),
                           'article_subject': doc.metadata.get('article_subject', ''),
                           'search_type': 'metadata'
                       }
                   })
           
           # 컨텐츠 결과 처리
           for doc in content_results:
               if doc.page_content not in seen_contents:
                   seen_contents.add(doc.page_content)
                   all_results.append({
                       'content': doc.page_content,
                       'metadata': {
                           'law_title': doc.metadata.get('law_title', ''),
                           'effective_date': doc.metadata.get('effective_date', ''),
                           'paragraph_number': doc.metadata.get('paragraph_number', ''),
                           'paragraph_subject': doc.metadata.get('paragraph_subject', ''),
                           'article_number': doc.metadata.get('article_number', ''),
                           'article_subject': doc.metadata.get('article_subject', ''),
                           'search_type': 'content'
                       }
                   })
           
           return all_results[:5]  # 상위 5개 결과 반환
           
       except Exception as e:
           raise Exception(f"법령 검색 중 오류 발생: {str(e)}")

   def _search_similar_qa(self, question: str) -> Optional[Dict[str, Any]]:
       """유사한 QA 쌍을 검색합니다."""
       try:
           similar_results = self.db.similarity_search(
               query=question,
               k=1
           )
           
           if similar_results:
               return {
                   "question": similar_results[0].page_content,
                   "answer": similar_results[0].metadata.get('answer', '')
               }
           return None
       except Exception as e:
           raise Exception(f"유사 QA 검색 중 오류 발생: {str(e)}")

   def _generate_answer(
       self,
       question: str,
       law_info: List[Dict[str, Any]],
       similar_qa: Optional[Dict[str, Any]]
   ) -> Dict[str, Any]:
       try:
           context = "\n".join([
               f"""
               법령명: {law['metadata']['law_title']}
               시행일자: {law['metadata']['effective_date']}
               조항번호: {law['metadata']['article_number']}
               조항제목: {law['metadata']['article_subject']}
               내용: {law['content']}
               """
               for law in law_info
           ])

           initial_prompt = self._format_prompt(context, question)
           raw_answer = self.llm.invoke(initial_prompt)
           current_answer = raw_answer.split('[답변]')[1].strip() if '[답변]' in raw_answer else raw_answer.strip()

           category = self.prompt_manager.classify_question(question)
           if category in ['registration', 'parking', 'pension']:
               current_answer = self._format_procedural_answer(current_answer)
           elif category in ['activity', 'child']:
               current_answer = self._format_service_answer(current_answer)
           elif category == 'welfare':
               current_answer = self._format_welfare_answer(current_answer)

           return {
               "answer": current_answer,
               "referenced_laws": law_info,
               "similar_qa_used": similar_qa is not None,
               "category": category
           }

       except Exception as e:
           raise Exception(f"답변 생성 중 오류 발생: {str(e)}")

   def _format_procedural_answer(self, answer: str) -> str:
       """절차 관련 답변 포맷팅"""
       sections = ["신청 자격", "필요 서류", "신청 절차", "처리 기간", "문의처"]
       formatted = []
       current_section = None
       
       for line in answer.split('\n'):
           for section in sections:
               if section in line:
                   current_section = section
                   formatted.append(f"\n[{section}]")
                   break
           else:
               if line.strip() and current_section:
                   formatted.append(f"- {line.strip()}")
       
       return "\n".join(formatted)

   def _format_service_answer(self, answer: str) -> str:
       """서비스 관련 답변 포맷팅"""
       sections = ["서비스 내용", "이용 방법", "지원 금액", "신청 방법", "문의처"]
       formatted = []
       current_section = None
       
       for line in answer.split('\n'):
           for section in sections:
               if section in line:
                   current_section = section
                   formatted.append(f"\n[{section}]")
                   break
           else:
               if line.strip() and current_section:
                   formatted.append(f"- {line.strip()}")
       
       return "\n".join(formatted)

   def _format_welfare_answer(self, answer: str) -> str:
       """복지 서비스 관련 답변 포맷팅"""
       sections = ["지원 내용", "신청 자격", "지원 금액", "신청 방법", "문의처"]
       formatted = []
       current_section = None
       
       for line in answer.split('\n'):
           for section in sections:
               if section in line:
                   current_section = section
                   formatted.append(f"\n[{section}]")
                   break
           else:
               if line.strip() and current_section:
                   formatted.append(f"- {line.strip()}")
       
       return "\n".join(formatted)

   def answer_question(self, question: str) -> Dict[str, Any]:
       """질문에 대한 답변을 생성합니다."""
       try:
           self.prompt_template = self._load_template(question)
           
           relevant_laws = self._search_relevant_laws(question)
           similar_qa = self._search_similar_qa(question)
           response = self._generate_answer(question, relevant_laws, similar_qa)
           
           response['template_category'] = self.prompt_manager.classify_question(question)
           
           return response
           
       except Exception as e:
           logging.error(f"질문 답변 중 오류 발생: {str(e)}")
           raise Exception(f"질문 답변 중 오류 발생: {str(e)}")

   def debug_qa_system(self, question: str):
       """검색 결과를 확인합니다."""
       print("\n=== 메타데이터 활용 검색 결과 확인 ===")
       results = self._search_relevant_laws(question)
       category = self.prompt_manager.classify_question(question)
       print(f"\n선택된 카테고리: {category}")
       
       for i, result in enumerate(results):
           print(f"\n[검색 결과 {i+1}]")
           print("법령명:", result['metadata']['law_title'])
           print("장 번호:", result['metadata']['paragraph_number'])
           print("장 제목:", result['metadata']['paragraph_subject'])
           print("조문번호:", result['metadata']['article_number'])
           print("조문제목:", result['metadata']['article_subject'])
           print("\n내용:")
           print(result['content'])

def setup_qa_system(
   custom_llm,
   custom_embeddings,
   custom_db,
   template_path: str
) -> LegalQASystem:
   """QA 시스템을 초기화하고 반환합니다."""
   return LegalQASystem(
       custom_llm=custom_llm,
       custom_embeddings=custom_embeddings,
       custom_db=custom_db,
       template_path=template_path
   )