import yaml
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

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
            return template.format(
                context=context,
                question=question
            )
        except KeyError:
            raise KeyError("템플릿에 'template' 키가 없습니다.")
        except Exception as e:
            raise ValueError(f"프롬프트 포맷팅 중 오류 발생: {str(e)}")
    
    def _search_relevant_laws(self, question: str) -> List[Dict[str, Any]]:
        """다중 쿼리로 관련 법령을 검색합니다."""
        try:
            # 검색 키워드 확장
            base_keywords = question.split()[:3]  # 원본 질문의 주요 키워드
            search_queries = [
                question,  # 원래 질문
                f"{base_keywords[0]} 대상자",  # 대상자 관점
                f"{base_keywords[0]} 자격 요건",  # 자격 관점
                f"{base_keywords[0]} 신청 자격"  # 신청 자격 관점
            ]
            
            all_results = []
            for query in search_queries:
                results = self.db.similarity_search(
                    query=query,
                    k=2  # 각 쿼리당 상위 2개만 검색
                )
                all_results.extend(results)
            
            # 중복 제거
            unique_results = []
            seen_contents = set()
            for doc in all_results:
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    unique_results.append({
                        'content': doc.page_content,
                        'metadata': {
                            'law_title': doc.metadata.get('law_title', ''),
                            'effective_date': doc.metadata.get('effective_date', ''),
                            'article_number': doc.metadata.get('article_number', ''),
                            'article_subject': doc.metadata.get('article_subject', '')
                        }
                    })
            
            return unique_results[:3]  # 상위 3개 결과 반환
            
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
            
            # 기본 프롬프트로 첫 답변 생성
            initial_prompt = self._format_prompt(context, question)
            raw_answer = self.llm.invoke(initial_prompt)
            
            # 답변 정제 - '[답변]' 태그가 없는 경우에도 처리
            if '[답변]' in raw_answer:
                current_answer = raw_answer.split('[답변]')[1].strip()
            else:
                current_answer = raw_answer.strip()
                
            # 불필요한 텍스트 제거
            if '다음은 장애인복지 관련 문의에 대한 답변을 생성하기 위한 지침입니다.' in current_answer:
                current_answer = current_answer.split('다음은 장애인복지 관련 문의에 대한 답변을 생성하기 위한 지침입니다.')[1].strip()
                
            if '답변 작성 시 주의사항:' in current_answer:
                current_answer = current_answer.split('답변 작성 시 주의사항:')[0].strip()
                
            # 유사 QA가 있는 경우 답변 보완
            if similar_qa and similar_qa.get('answer'):
                supplementary_prompt = f"""
                    다음 답변을 참고하여 기존 답변을 보완해주세요:
                    기존 질문: {similar_qa['question']}
                    기존 답변: {similar_qa['answer']}
                    현재 답변: {current_answer}
                    
                    답변 형식:
                    1. 신청 자격 요건
                    2. 제외 대상
                    3. 예외사항
                    4. 추가 안내사항
                """
                enhanced_answer = self.llm.invoke(supplementary_prompt)
                
                # 보완된 답변에서도 불필요한 부분 제거
                if '[답변]' in enhanced_answer:
                    final_answer = enhanced_answer.split('[답변]')[1].strip()
                else:
                    final_answer = enhanced_answer.strip()
            else:
                final_answer = current_answer
    
            return {
                "answer": final_answer,
                "referenced_laws": law_info,
                "similar_qa_used": similar_qa is not None
            }
            
        except Exception as e:
            raise Exception(f"답변 생성 중 오류 발생: {str(e)}")
    
    def _extract_answer_points(self, answer: str) -> Dict[str, List[str]]:
        """답변에서 각 항목별 포인트를 추출합니다."""
        categories = {
            "신청 자격 요건": [],
            "제외 대상": [],
            "예외사항": [],
            "추가 안내사항": []
        }
        
        current_category = None
        for line in answer.split('\n'):
            line = line.strip()
            if any(category in line for category in categories.keys()):
                current_category = next(cat for cat in categories.keys() if cat in line)
            elif line.startswith('-') and current_category:
                point = line[1:].strip()
                if point and point not in categories[current_category]:
                    categories[current_category].append(point)
        
        return categories
    
    def _merge_answers(self, current_points: Dict[str, List[str]], similar_points: Dict[str, List[str]]) -> str:
        """두 답변의 포인트를 병합하여 최종 답변을 생성합니다."""
        merged_points = {}
        
        # 각 카테고리별로 포인트 병합
        for category in current_points.keys():
            merged_points[category] = list(set(current_points[category]))
            if category in similar_points:
                # 유사 답변의 포인트 중 현재 답변에 없는 것만 추가
                for point in similar_points[category]:
                    if not any(self._is_similar_point(point, existing) for existing in merged_points[category]):
                        merged_points[category].append(point)
        
        # 병합된 포인트로 답변 생성
        final_answer = []
        for category, points in merged_points.items():
            if points:  # 해당 카테고리에 포인트가 있는 경우만 포함
                final_answer.append(f"{category}")
                for point in points:
                    final_answer.append(f"- {point}")
                final_answer.append("")  # 카테고리 간 빈 줄 추가
        
        return "\n".join(final_answer).strip()
    
    def _is_similar_point(self, point1: str, point2: str) -> bool:
        """두 포인트가 유사한지 확인합니다."""
        # 간단한 유사도 체크 (필요에 따라 더 정교한 방법 사용 가능)
        return (
            point1.lower() in point2.lower() or 
            point2.lower() in point1.lower() or
            self._calculate_similarity(point1, point2) > 0.8
        )
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """두 텍스트의 유사도를 계산합니다."""
        # 여기에 텍스트 유사도 계산 로직 구현
        # 예: 코사인 유사도, 레벤슈타인 거리 등 사용
        # 임시로 간단한 구현
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0.0
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """질문에 대한 답변을 생성합니다."""
        try:
            relevant_laws = self._search_relevant_laws(question)
            similar_qa = self._search_similar_qa(question)
            response = self._generate_answer(question, relevant_laws, similar_qa)
            return response
        except Exception as e:
            raise Exception(f"질문 답변 중 오류 발생: {str(e)}")

    def _search_relevant_laws(self, question: str) -> List[Dict[str, Any]]:
        """메타데이터와 컨텐츠를 결합한 검색을 수행합니다."""
        try:
            # 1. 메타데이터 필터 조건 설정
            metadata_filters = {
                "article_subject": [
                    "신청", "절차", "방법", "서류", "등록", "신고",
                    "제출", "접수", "처리", "기준", "요건"
                ]
            }
            
            # 2. 메타데이터 기반 검색
            metadata_results = self.db.similarity_search(
                query=question,
                k=5,
                filter={"article_subject": {"$in": metadata_filters["article_subject"]}}
            )
            
            # 3. 일반 컨텐츠 기반 검색
            content_results = self.db.similarity_search(
                query=question,
                k=5
            )
            
            # 4. 결과 병합 및 정렬
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
    
    def debug_qa_system(self, question: str):
        """검색 결과를 확인합니다."""
        print("\n=== 메타데이터 활용 검색 결과 확인 ===")
        results = self._search_relevant_laws(question)
        
        for i, result in enumerate(results):
            print(f"\n[검색 결과 {i+1}] (검색 타입: {result['metadata']['search_type']})")
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