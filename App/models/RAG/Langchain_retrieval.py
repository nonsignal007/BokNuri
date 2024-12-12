from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
import numpy as np
import logging

@dataclass
class SearchResult:
    content: str
    metadata: Dict[str, Any]
    score: float
    source: str  # 'law' 또는 'qa'

class HybridSearchEngine:
    def __init__(self, embeddings, law_db, qa_db):
        self.embeddings = embeddings
        self.law_db = law_db
        self.qa_db = qa_db
        
        # BM25 관련 변수들
        self.law_bm25 = None
        self.qa_bm25 = None
        
        self.law_corpus = []
        self.qa_corpus = []
        
        self.law_metadata = []
        self.qa_metadata = []
    
    def initialize_bm25(self, law_documents: List[dict], qa_documents: List[dict]):
        """법령과 QA DB의 BM25 초기화"""
        # 법령 문서 초기화
        self.law_corpus = [doc['content'] for doc in law_documents]
        self.law_metadata = [doc['metadata'] for doc in law_documents]
        tokenized_law_corpus = [doc.split() for doc in self.law_corpus]
        self.law_bm25 = BM25Okapi(tokenized_law_corpus)
        
        # QA 문서 초기화
        self.qa_corpus = [doc['content'] for doc in qa_documents]
        self.qa_metadata = [doc['metadata'] for doc in qa_documents]
        tokenized_qa_corpus = [doc.split() for doc in self.qa_corpus]
        self.qa_bm25 = BM25Okapi(tokenized_qa_corpus)

    def search_law(self, query: str, k_dense: int = 3, k_bm25: int = 2) -> List[SearchResult]:
        """법령 DB 검색"""
        results = []
        
        # Dense Retrieval
        dense_results = self.law_db.similarity_search_with_score(query, k=k_dense)
        for doc, score in dense_results:
            results.append(SearchResult(
                content=doc.page_content,
                metadata=doc.metadata,
                score=float(score),
                source='law'
            ))
            
        # BM25
        tokenized_query = query.split()
        bm25_scores = self.law_bm25.get_scores(tokenized_query)
        top_k = np.argsort(bm25_scores)[-k_bm25:][::-1]
        
        for idx in top_k:
            results.append(SearchResult(
                content=self.law_corpus[idx],
                metadata=self.law_metadata[idx],
                score=float(bm25_scores[idx]),
                source='law'
            ))
            
        return results
    
    def search_qa(self, query: str, k_dense: int = 3, k_bm25: int = 2) -> List[SearchResult]:
        """QA DB 검색"""
        results = []
        
        # Dense Retrieval
        dense_results = self.qa_db.similarity_search_with_score(query, k=k_dense)
        for doc, score in dense_results:
            results.append(SearchResult(
                content=doc.page_content,
                metadata=doc.metadata,
                score=float(score),
                source='qa'
            ))
            
        # BM25
        tokenized_query = query.split()
        bm25_scores = self.qa_bm25.get_scores(tokenized_query)
        top_k = np.argsort(bm25_scores)[-k_bm25:][::-1]
        
        for idx in top_k:
            results.append(SearchResult(
                content=self.qa_corpus[idx],
                metadata=self.qa_metadata[idx],
                score=float(bm25_scores[idx]),
                source='qa'
            ))
            
        return results
    
    def hybrid_search(self, query: str, k_dense: int = 3, k_bm25: int = 2) -> Dict[str, List[SearchResult]]:
        """법령과 QA 하이브리드 검색 수행"""
        # 각 DB 검색
        law_results = self.search_law(query, k_dense, k_bm25)
        qa_results = self.search_qa(query, k_dense, k_bm25)
        
        # 각각의 결과를 점수순으로 정렬
        law_results.sort(key=lambda x: x.score, reverse=True)
        qa_results.sort(key=lambda x: x.score, reverse=True)
        
        return {
            'law_results': law_results[:3],      # 상위 3개
            'qa_results': qa_results[:3]         # 상위 3개
        }

def format_laws(results: List[SearchResult]) -> str:
    """Format law search results in English"""
    return "\n".join([
        f"""
        Law Title: {result.metadata.get('law_title', 'Unknown')}
        Article Number: {result.metadata.get('article_number', 'Unknown')}
        Article Subject: {result.metadata.get('article_subject', 'Unknown')}
        Content: {result.content}
        """
        for result in results
    ])

def format_qna(results: List[SearchResult]) -> str:
    """Format QA search results in English"""
    return "\n".join([
        f"""
        Q: {result.metadata.get('question', 'Unknown')}
        A: {result.content}
        """
        for result in results
    ])

class LegalQASystem:
    def __init__(self, custom_llm, custom_embeddings, law_db, qa_db):
        self.llm = custom_llm
        self.search_engine = HybridSearchEngine(
            custom_embeddings,
            law_db,
            qa_db
        )
        
    def _format_retrieval_results(self, search_results: Dict[str, List[SearchResult]]) -> str:
        """Format retrieval results with scores for better context"""
        formatted_results = []
        
        # Process law results
        law_results = search_results['law_results']
        if law_results:
            formatted_results.append("Most Relevant Laws (by similarity score):")
            for i, result in enumerate(law_results, 1):
                formatted_results.append(f"""
                Result #{i} (Score: {result.score:.3f})
                Law: {result.metadata.get('law_title', 'Unknown')}
                Article: {result.metadata.get('article_number', 'Unknown')}
                Content: {result.content}
                """.strip())
        
        # Process QA results
        qa_results = search_results['qa_results']
        if qa_results:
            formatted_results.append("\nMost Relevant Previous Cases (by similarity score):")
            for i, result in enumerate(qa_results, 1):
                formatted_results.append(f"""
                Result #{i} (Score: {result.score:.3f})
                Q: {result.metadata.get('question', 'Unknown')}
                A: {result.content}
                """.strip())
                
        return "\n".join(formatted_results)

def generate_answer(self, question: str, search_results: Dict[str, List[SearchResult]], user_info: Optional[Dict] = None) -> Dict[str, Any]:
    """검색 결과를 바탕으로 답변 생성"""
    try:
        # 검색 결과 포맷팅
        retrieval_context = self._format_retrieval_results(search_results)
        
        # 시스템 프롬프트
        system_prompt = """당신은 시각장애인을 위한 복지 서비스 상담사입니다. 다음 지침을 따라주세요:
1. 반드시 한국어로 답변해주세요
2. 5줄 이내로 간단명료하게 답변해주세요
3. 검색 점수가 높은 정보를 우선적으로 참고해주세요
4. 답변은 다음 형식으로 구성해주세요:
   - 핵심 답변 요약
   - 구체적인 지원 내용
   - 신청 방법 및 연락처
5. 존댓말을 사용해 정중하게 답변해주세요
6. 검색 점수가 가장 높은 결과를 기반으로 답변해주세요"""

        # 사용자 정보 컨텍스트
        user_context = ""
        if user_info:
            user_context = f"""신청자 정보:
- 장애등급: {user_info.get('disability_grade', '미상')}
- 지역: {user_info.get('region', '미상')}
- 연령: {user_info.get('age', '미상')}
- 기초생활보장: {user_info.get('basic_living_security', '미상')}"""

        # 검색 컨텍스트와 함께 메시지 구성
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"""
{user_context}

질문: {question}

검색 결과 및 관련성 점수:
{retrieval_context}

위의 검색 결과를 바탕으로 (높은 점수의 결과를 우선 참고하여) 5줄 이내로 간단명료하게 한국어로 답변해주세요."""
            }
        ]

        # 응답 생성
        response = self.llm.invoke(messages)
        
        return {
            "answer": response.strip(),
            "search_results": [
                {
                    "content": r.content,
                    "metadata": r.metadata,
                    "score": r.score,
                    "source": r.source
                }
                for r in search_results['law_results'] + search_results['qa_results']
            ]
        }

    except Exception as e:
        logging.error(f"답변 생성 중 오류 발생: {str(e)}")
        return {
            "answer": "죄송합니다. 답변 생성 중 오류가 발생했습니다.",
            "search_results": []
        }

    def answer_question(self, question: str, user_info: Optional[Dict] = None) -> Dict[str, Any]:
        """Main method to answer questions by combining search and answer generation"""
        try:
            # First, perform hybrid search
            search_results = self.search_engine.hybrid_search(question)
            
            # Then, generate answer using the search results
            response = self.generate_answer(question, search_results, user_info)
            
            return response
            
        except Exception as e:
            logging.error(f"Error in answer_question: {str(e)}")
            return {
                "answer": "죄송합니다. 질문 처리 중 오류가 발생했습니다.",
                "search_results": []
            }