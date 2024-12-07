import os
import yaml
import streamlit as st
from audio_recorder_streamlit import audio_recorder

from models.STT.STT_models import STTModel
from models.RAG.Langchain import load_rag_model  


class Config:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)

    def load_config(self, path: str):
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def get(self, key):
        return self.config[key]

class VoiceBot:
    def __init__(self):
        self.config = Config('config.yaml')

        # session_state 초기화
        if 'is_recording' not in st.session_state:
            st.session_state.is_recording = False
        if 'stt_model' not in st.session_state:
            st.session_state.stt_model = self.load_stt_model(self.config)

        # RAG 체인 초기화
        self.rag_chain = load_rag_model('/workspace/LangEyE/langchain_folder/pdf/장애인ㆍ노인ㆍ임산부 등의 편의증진 보장에 관한 법률 시행규칙.pdf')

        st.set_page_config(
            page_title="시각장애인을 위한 보이스 봇",
            page_icon="🎤",
            layout="centered"
        )

    def load_stt_model(self, config):
        """STT 모델 로드"""
        model = STTModel(config)
        return model

    def speech_to_text(self, audio_bytes):
        """STT: 음성을 텍스트로 변환"""
        model = st.session_state.stt_model
        return model.transcribe(audio_bytes)

    
    def recording(self, col2):
        """오디오 녹음 및 처리"""
        audio_bytes = audio_recorder(
            text="녹음 버튼을 눌러 질문하세요.",
            recording_color="#e87474",
            neutral_color="#6aa36f"
        )
        
        if audio_bytes:
            st.success("녹음이 완료되었습니다!")
            st.audio(audio_bytes, format="audio/wav")
            try:
                # STT: 음성을 텍스트로 변환
                question = self.speech_to_text(audio_bytes)
                st.write("**음성 인식 결과:**", question)

                # RAG: 질문에 대한 답변 생성
                st.info("답변 생성 중...")
                response = self.rag_chain.invoke(question)
                st.write("**생성된 답변:**", response)

            except Exception as e:
                st.error(f"오류 발생: {e}")

    def run(self):
        """앱 실행"""
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.title("STT 기반 RAG 보이스 봇")
            self.recording(col2)


if __name__ == "__main__":
    print(os.getcwd())
    VoiceBot().run()
