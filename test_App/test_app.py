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
        self.config = Config('test_App/config.yaml')
        
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

        # CSS 스타일 추가
        st.markdown("""
            <style>
            /* Streamlit 기본 패딩 제거 */
            .block-container {
                padding: 0;
            }
            
            /* 오디오 레코더 컨테이너 중앙 정렬 */
            .element-container {
                position: fixed;
                top: 50%;
                left: 72%;
                transform: translate(-50%, -50%);
                margin: 0 !important;
            }
            
            /* 오디오 녹음 버튼 스타일링 */
            .audio-recorder {
                width: 200px !important;
                height: 200px !important;
                border-radius: 50% !important;
                background-color: #FF4B4B !important;
                border: none !important;
                display: flex !important;
                justify-content: center !important;
                align-items: center !important;
                padding: 0 !important;
                cursor: pointer !important;
                margin: 0 auto !important;
            }

            .audio-recorder:hover {
                background-color: #FF0000 !important;
            }

            .audio-recorder:active {
                background-color: #CC0000 !important;
            }

            /* 마이크 아이콘 스타일 */
            .audio-recorder svg {
                width: 100px !important;
                height: 100px !important;
                fill: white !important;
            }

            /* 오디오 플레이어 스타일 */
            .stAudio {
                position: fixed;
                left: 50%;
                transform: translateX(-50%);
                bottom: 20px;
                width: auto;
            }
            
            .stAudio > div {
                display: flex;
                justify-content: center;
            }
            </style>
        """, unsafe_allow_html=True)
    
    def load_stt_model(self, config):
        model = STTModel(config)
        return model
        
    def speech_to_text(self, audio_bytes):
        """음성을 텍스트로 변환"""
        model = st.session_state.stt_model
        return model.transcribe(audio_bytes)
        
    def recording(self):
        # 오디오 녹음 처리
        audio_bytes = audio_recorder(
            text="",
            recording_color="#e87474",
            neutral_color="#6aa36f",
            icon_size="6x"  # 마이크 아이콘 크기 증가
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
        self.recording()

if __name__ == "__main__":
    print(os.getcwd())
    VoiceBot().run()