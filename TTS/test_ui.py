import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import os
import tempfile
from datetime import datetime
import time

def text_to_speech(text, lang='ko'):
    """텍스트를 음성으로 변환하는 함수"""
    tts = gTTS(text=text, lang=lang)
    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        tts.save(fp.name)
        return fp.name

def speech_to_text():
    """음성을 텍스트로 변환하는 함수"""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("말씀해주세요...")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio, language='ko-KR')
            return text
        except sr.UnknownValueError:
            return "죄송합니다. 음성을 인식하지 못했습니다."
        except sr.RequestError:
            return "죄송합니다. 음성 인식 서비스에 문제가 발생했습니다."

def main():
    st.title("시각장애인을 위한 음성 도우미")
    
    # 세션 상태 초기화
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # 음성 입력 버튼
    if st.button("음성으로 말하기"):
        user_input = speech_to_text()
        if user_input:
            # 사용자 입력 저장
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # 간단한 응답 로직
            if "안녕" in user_input:
                response = "안녕하세요! 무엇을 도와드릴까요?"
            elif "날씨" in user_input:
                response = "오늘은 맑은 날씨입니다."
            elif "시간" in user_input:
                response = f"현재 시간은 {datetime.now().strftime('%H시 %M분')}입니다."
            else:
                response = "죄송합니다. 다시 한 번 말씀해 주시겠어요?"
            
            # 응답 저장
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # 음성으로 응답 재생
            audio_file = text_to_speech(response)
            st.audio(audio_file)
            os.unlink(audio_file)  # 임시 파일 삭제
    
    # 대화 기록 표시
    for message in st.session_state.messages:
        role = "사용자" if message["role"] == "user" else "도우미"
        st.write(f"{role}: {message['content']}")

if __name__ == "__main__":
    main()
