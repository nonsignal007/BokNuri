import streamlit as st
from models.RAG.Langchain import load_rag_model  # Langchain.py에서 모델 가져오기

# Streamlit 설정
st.set_page_config(page_title="RAG 기반 Q&A", layout="wide")
st.title("RAG 기반 Q&A 시스템")


# RAG 모델 로드 (세션 상태를 활용)
if "rag_chain" not in st.session_state:
    try:
        st.info("모델 로드 중... (한 번만 실행됩니다.)")
        st.session_state.rag_chain = load_rag_model()
        st.success("모델 로드 완료!")
    except Exception as e:
        st.error(f"모델 로드 중 오류가 발생했습니다: {e}")

# 질문 입력
question = st.text_input("질문을 입력하세요:")

# 답변 생성
if st.button("답변 생성"):
    if "rag_chain" not in st.session_state:
        st.error("모델이 로드되지 않았습니다.")
    elif not question:
        st.error("질문을 입력하세요.")
    else:
        try:
            # 질문 처리
            st.info("질문 처리 중...")
            response = st.session_state.rag_chain.invoke(question)

            # 결과 출력
            st.markdown(f"### 질문: {question}")
            st.markdown(f"### 답변: {response}")
        except Exception as e:
            st.error(f"질문 처리 중 오류가 발생했습니다: {e}")
