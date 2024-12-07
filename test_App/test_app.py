import yaml
import os
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from models.STT.STT_models import STTModel

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

        st.set_page_config(
            page_title="시각장애인을 위한 보이스 봇",
            page_icon="🎤",
            layout="centered"
        )
    
    def load_stt_model(self, config):
        model = STTModel(config)
        return model
        
    def speech_to_text(self, audio_bytes):
        """음성을 텍스트로 변환"""
        # TODO: STT 모델을 사용하여 음성을 텍스트로 변환
        # 예: text = self.stt_model.transcribe(audio_bytes)
        # return text
        model = st.session_state.stt_model
        return model.transcribe(audio_bytes)
        
    def recording(self, col2):
        # 오디오 녹음 처리
        audio_bytes = audio_recorder(
            text="",
            recording_color="#e87474",
            neutral_color="#6aa36f"
        )
        
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")

            try:
                # STT 모델을 사용하여 음성을 텍스트로 변환
                text = self.speech_to_text(audio_bytes)
                print("음성 인식 결과:", text)
                
            except Exception as e:
                print(f"Error processing audio: {str(e)}")
                
    def run(self):
        col1, col2, col3 = st.columns([1,2,1])

        with col2:
            st.title("보이스 봇")
            self.recording(col2)

if __name__ == "__main__":
    print(os.getcwd())
    VoiceBot().run()