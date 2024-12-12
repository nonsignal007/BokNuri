import yaml
from typing import List, Dict, Any, Optional
from pathlib import Path

class LegalQASystem:
    def __init__(
        self,
        custom_llm,
        custom_embeddings,
        custom_db,
        template_path: str = "templates/qa_prompt.yaml"
    ):
        """
        LegalQASystem 초기화
        
        Args:
            custom_llm: HuggingFacePipeline 인스턴스
            custom_embeddings: HuggingFaceEmbeddings 인스턴스
            custom_db: Chroma 데이터베이스 인스턴스
            template_path: 프롬프트 템플릿 YAML 파일 경로
        """
        self.llm = custom_llm
        self.embeddings = custom_embeddings
        self.db = custom_db
        self.prompt_template = self._load_template(template_path)
    
    def _load_template(self, template_path: str) -> dict:
        """YAML 파일에서 프롬프트 템플릿을 로드합니다."""
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template_data = yaml.safe_load(f)
            return template_data
        except FileNotFoundError:
            raise FileNotFoundError(f"템플릿 파일을 찾을 수 없습니다: {template_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"YAML 파일 파싱 중 오류 발생: {str(e)}")
    
    def _format_prompt(self, context: str, question: str) -> str:
        """템플릿을 사용하여 프롬프트를 포맷팅합니다."""
        try:
            template = self.prompt_template['template']
            formatted_prompt = template.format(
                context=context,
                question=question
            )
            return formatted_prompt
        except KeyError:
            raise KeyError("템플릿에 'template' 키가 없습니다.")
        except Exception as e:
            raise ValueError(f"프롬프트 포맷팅 중 오류 발생: {str(e)}")

    def _search_relevant_laws(self, question: str) -> List[Dict[str, Any]]:
        """메타데이터와 컨텐츠를 결합한 검색을 수행합니다."""
        try:
            # 메타데이터 필터 조건 설정
            metadata_filters = {
                "article_subject": [
                    "신청", "절차", "방법", "서류", "등록", "신고",
                    "제출", "접수", "처리", "기준", "요건"
                ]
            }
            
            # 메타데이터 기반 검색
            metadata_results = self.db.similarity_search(
                query=question,
                k=5,
                filter={"article_subject": {"$in": metadata_filters["article_subject"]}}
            )
            
            # 일반 컨텐츠 기반 검색
            content_results = self.db.similarity_search(
                query=question,
                k=5
            )
            
            # 결과 병합 및 중복 제거
            all_results = []
            seen_contents = set()
            
            for doc in metadata_results + content_results:
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    all_results.append({
                        'content': doc.page_content,
                        'metadata': {
                            'law_title': doc.metadata.get('law_title', ''),
                            'effective_date': doc.metadata.get('effective_date', ''),
                            'article_number': doc.metadata.get('article_number', ''),
                            'article_subject': doc.metadata.get('article_subject', ''),
                            'search_type': 'metadata' if doc in metadata_results else 'content'
                        }
                    })
            
            return all_results[:5]  # 상위 5개 결과 반환
            
        except Exception as e:
            raise Exception(f"법령 검색 중 오류 발생: {str(e)}")
    
    def _generate_answer(
        self,
        question: str,
        law_info: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """답변을 생성합니다."""
        try:
            # 컨텍스트 구성
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
            
            # 프롬프트 생성 및 답변 획득
            prompt = self._format_prompt(context, question)
            raw_answer = self.llm(prompt)
            
            if isinstance(raw_answer, list):
                raw_answer = raw_answer[0]['generated_text']
            
            # 답변 정제
            if '[답변]' in raw_answer:
                answer = raw_answer.split('[답변]')[1].strip()
            else:
                answer = raw_answer.strip()
            
            # 불필요한 텍스트 제거
            unwanted_texts = [
                '다음은 장애인복지 관련 문의에 대한 답변을 생성하기 위한 지침입니다.',
                '답변 작성 시 주의사항:'
            ]
            for text in unwanted_texts:
                if text in answer:
                    answer = answer.split(text)[0].strip()
            
            return {
                "answer": answer,
                "referenced_laws": law_info
            }
            
        except Exception as e:
            raise Exception(f"답변 생성 중 오류 발생: {str(e)}")
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """질문에 대한 답변을 생성합니다."""
        try:
            relevant_laws = self._search_relevant_laws(question)
            response = self._generate_answer(question, relevant_laws)
            return response
        except Exception as e:
            raise Exception(f"질문 답변 중 오류 발생: {str(e)}")