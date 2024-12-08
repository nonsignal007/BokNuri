import yaml
import os
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from models.STT.STT_models import STTModel
# from models.TTS.TTS_models import TTSModel  # TTS 모델 import 추가

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
        if 'tts_model' not in st.session_state:  # TTS 모델 세션 상태 추가
            st.session_state.tts_model = self.load_tts_model(self.config)
        if 'conversation_history' not in st.session_state:  # 대화 기록 추가
            st.session_state.conversation_history = []

        st.set_page_config(
            page_title="시각장애인을 위한 보이스 봇",
            page_icon="🎤",
            layout="centered"
        )

        # CSS 스타일 (이전과 동일)
        st.markdown("""
            <style>
            .block-container {
                padding: 0;
            }
            
            .element-container {
                position: fixed;
                top: 50%;
                left: 72%;
                transform: translate(-50%, -50%);
                margin: 0 !important;
            }
            
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

            .audio-recorder svg {
                width: 100px !important;
                height: 100px !important;
                fill: white !important;
            }

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
    
    def load_tts_model(self, config):  # TTS 모델 로드 함수 추가
        # model = TTSModel(config)
        # return model
        pass
        
    def speech_to_text(self, audio_bytes):
        """음성을 텍스트로 변환"""
        model = st.session_state.stt_model
        return model.transcribe(audio_bytes)
    
    def text_to_speech(self, text):  # TTS 함수 추가
        """텍스트를 음성으로 변환"""
        model = st.session_state.tts_model
        return model.synthesize(text)
    
    def process_conversation(self, user_input):
        """사용자 입력을 처리하고 응답 생성"""
        # 여기에 챗봇 로직 구현
        # 예시로 간단한 에코 응답
        response = f"입력하신 내용은: {user_input}"
        return response
        
    def recording(self):
        # 오디오 녹음 처리
        audio_bytes = audio_recorder(
            text="",
            recording_color="#e87474",
            neutral_color="#6aa36f",
            icon_size="6x"
        )
        
        if audio_bytes:
            # 사용자 음성 재생
            st.audio(audio_bytes, format="audio/wav")

            try:
                # STT로 음성을 텍스트로 변환
                user_text = self.speech_to_text(audio_bytes)
                print("음성 인식 결과:", user_text)
                
                # 대화 처리
                response_text = self.process_conversation(user_text)
                
                # TTS로 응답을 음성으로 변환
                response_audio = self.text_to_speech(response_text)
                
                # 응답 음성 재생
                st.audio(response_audio, format="audio/wav")
                
                # 대화 기록 저장
                st.session_state.conversation_history.append({
                    "user": user_text,
                    "bot": response_text
                })
                
            except Exception as e:
                print(f"Error processing audio: {str(e)}")
                
    def run(self):
        self.recording()

if __name__ == "__main__":
    print(os.getcwd())
    VoiceBot().run()