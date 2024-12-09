import gradio as gr
import speech_recognition as sr
import scipy.io.wavfile as wavfile
import tempfile
import os
import time
import re
from gtts import gTTS
import numpy as np

from models.RAG.Langchain import load_rag_model  # Langchain.py에서 모델 가져오기


########## STT , TTS ###############
def speech_to_text(audio):
    """음성을 텍스트로 변환"""
    if audio is None:
        return None
    
    sample_rate, data = audio
    recognizer = sr.Recognizer()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
        wavfile.write(temp_audio.name, sample_rate, data)
        
        with sr.AudioFile(temp_audio.name) as source:
            audio_data = recognizer.record(source)
            
        try:
            text = recognizer.recognize_google(audio_data, language='ko-KR')
            return text
        except Exception as e:
            return None
        finally:
            os.unlink(temp_audio.name)

def text_to_speech(text):
    """텍스트를 음성으로 변환하는 함수"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            tts = gTTS(text=text, lang='ko')
            tts.save(fp.name)
            return fp.name
    except Exception as e:
        return f"음성 변환 오류: {str(e)}"


########## Switch Tab ###############

def check_command(text):
    """명령어 확인"""
    if text is None:
        return None
    elif "설정" in text:
        return "settings"
    elif "음성" in text:
        return "voice"
    return None

def process_voice(audio, current_mode):
    """음성 입력 처리"""
    if audio is None:
        return current_mode, gr.update(), gr.update(), None
    
    text = speech_to_text(audio)
    command = check_command(text)
    
    if command:
        voice_visible = command == "voice"
        settings_visible = command == "settings"
        
        # 설정 모드로 전환될 때만 설정 안내 음성 생성
        settings_audio = None
        if settings_visible:
            setting_text = "안녕하세요. 복지 서비스 설정 페이지 입니다. 이름, 성별, 나이, 소득분위, 장애등급을 말씀해주세요"
            settings_audio = text_to_speech(setting_text)
        
        return command, \
               gr.update(visible=voice_visible), \
               gr.update(visible=settings_visible), \
               settings_audio
    
    return current_mode, gr.update(), gr.update(), None

########## Personal Setting ###############
    
def extract_settings(text):
    """사용자 음성에서 설정 정보 추출"""
    if text is None:
        return None, None, None, None, None
    
    # 소득분위 추출 (1~10분위)
    income_match = re.search(r'(\d+)\s*분위', text)
    income_level = income_match.group(1) if income_match else None
    
    # 장애등급 추출 (1~6급)
    disability_match = re.search(r'(\d+)\s*(?:급|등급)', text)
    disability_grade = disability_match.group(1) if disability_match else None
    
    # 나이 추출
    age_match = re.search(r'(\d+)\s*(?:살|세)', text)
    age = age_match.group(1) if age_match else None
    
    # 성별 추출
    gender = None
    if any(keyword in text for keyword in ['남자', '남성', '남']):
        gender = '남성'
    elif any(keyword in text for keyword in ['여자', '여성', '여']):
        gender = '여성'
    
    # 이름 추출 (자연스러운 발화 패턴 반영)
    # "나는 홍길동이야", "홍길동입니다", "홍길동이에요" 등의 패턴 포함
    name_patterns = [
        r'나는\s*([가-힣]+)(?:이야|야|이에요|예요|입니다)',
        r'([가-힣]+)(?:이야|야|이에요|예요|입니다)',
        r'이름은\s*([가-힣]+)',
        r'([가-힣]{2,4})(?:야|이야|이에요|예요|입니다)'
    ]
    
    name = None
    for pattern in name_patterns:
        name_match = re.search(pattern, text)
        if name_match:
            name = name_match.group(1)
            break
    
    return income_level, disability_grade, age, gender, name

def process_settings_input(audio, current_settings, current_mode):
    """설정 음성 입력 처리"""
    if audio is None:
        return current_settings, None, current_mode, gr.update(), gr.update()
    
    text = speech_to_text(audio)
    if text is None:
        return current_settings, None, current_mode, gr.update(), gr.update()
    
    # 확인 응답 체크
    if any(keyword in text for keyword in ['네', '예', '맞아', '맞습니다']):
        start_message = "이제부터 복지 서비스를 시작하겠습니다. 궁금하신점이 무엇인가요?"
        start_audio = text_to_speech(start_message)
        return current_settings, start_audio, "voice", gr.update(visible=False), gr.update(visible=True)
    
    income, disability, age, gender, name = extract_settings(text)
    
    # 현재 설정 업데이트
    new_settings = current_settings or {}
    if income: new_settings['income_level'] = income
    if disability: new_settings['disability_grade'] = disability
    if age: new_settings['age'] = age
    if gender: new_settings['gender'] = gender
    if name: new_settings['name'] = name
    
    # 설정 확인 음성 생성
    confirmation = f"입력하신 정보를 확인해드립니다. "
    if name: confirmation += f"{name}님, "
    if income: confirmation += f"소득분위 {income}분위, "
    if disability: confirmation += f"장애 {disability}등급, "
    if age: confirmation += f"나이 {age}세, "
    if gender: confirmation += f"{gender}이시네요. "
    
    confirmation += "입력하신 정보가 맞다면 '네', 수정이 필요하시다면 '아니오'라고 말씀해 주세요."
    
    confirmation_audio = text_to_speech(confirmation)
    
    return new_settings, confirmation_audio, current_mode, gr.update(), gr.update()

########## LangChain ###############
def format_user_info(settings):
    """사용자 설정 정보를 문자열로 포맷팅"""
    if not settings:
        return "설정된 정보가 없습니다."
    
    info_parts = []
    if 'name' in settings:
        info_parts.append(f"이름: {settings['name']}")
    if 'age' in settings:
        info_parts.append(f"나이: {settings['age']}세")
    if 'gender' in settings:
        info_parts.append(f"성별: {settings['gender']}")
    if 'income_level' in settings:
        info_parts.append(f"소득분위: {settings['income_level']}분위")
    if 'disability_grade' in settings:
        info_parts.append(f"장애등급: {settings['disability_grade']}급")
    
    return ", ".join(info_parts)

class ChatbotState:
    def __init__(self):
        """챗봇 상태 초기화"""
        self.create_chain = load_rag_model()
        self.chain = None
        self.settings = None
        self.initialize_chain()
    
    def initialize_chain(self):
        """기본 체인 초기화"""
        try:
            default_info = "설정되지 않은 사용자"
            self.chain = self.create_chain(default_info)
        except Exception as e:
            print(f"Chain initialization error: {str(e)}")
            self.chain = None

    def update_settings(self, settings):
        """사용자 설정 업데이트 및 체인 재생성"""
        try:
            self.settings = settings
            user_info = format_user_info(settings)
            self.chain = self.create_chain(user_info)
            return True
        except Exception as e:
            print(f"Settings update error: {str(e)}")
            return False

# 전역 챗봇 상태 객체 생성
chatbot_state = ChatbotState()
    
def process_voice_chat(audio, current_settings):
    """음성 채팅 처리"""
    if audio is None:
        return text_to_speech("음성 입력이 없습니다.")
    
    # 음성을 텍스트로 변환
    text = speech_to_text(audio)
    if text is None:
        return text_to_speech("죄송합니다. 음성을 인식하지 못했습니다.")
    
    try:
        # 설정이 변경되었다면 체인 업데이트
        if current_settings != chatbot_state.settings:
            chatbot_state.update_settings(current_settings)
        
        # RAG 체인으로 응답 생성
        if chatbot_state.chain is None:
            response = "죄송합니다. 시스템에 문제가 있습니다. 잠시 후 다시 시도해주세요."
        else:
            response = chatbot_state.chain.invoke(text)
        
        # 응답을 음성으로 변환
        return text_to_speech(response)
    
    except Exception as e:
        print(f"Error in process_voice_chat: {str(e)}")
        return text_to_speech("죄송합니다. 응답 생성 중 오류가 발생했습니다.")

########## Interface ###############
with gr.Blocks() as demo:
    current_mode = gr.State("voice")
    current_settings = gr.State(None)
    
    welcome_text = "안녕하세요. 서울특별시 구로구의 시각장애인들을 위한 보이스 복지 서비스입니다. 설정하신 정보에 맞게 복지 정보를 알려드릴게요. 만약 설정하신게 없으시다면 '설정'을 말씀해 주세요"
    welcome_audio = text_to_speech(welcome_text)
    gr.Audio(welcome_audio, autoplay=True, visible=False)
    
    with gr.Row():
        gr.Markdown("# 복지 Q&A Voice Bot")
    
    with gr.Row():
        main_audio_input = gr.Audio(sources=["microphone"], type="numpy")
    
    with gr.Row(visible=True) as voice_section:
        with gr.Column():
            gr.Markdown("## 복지 서비스")
            voice_chat_input = gr.Audio(sources=["microphone"], type="numpy")
            voice_chat_output = gr.Audio(label="AI 응답")
    
    with gr.Row(visible=False) as settings_section:
        with gr.Column():
            gr.Markdown("## 설정")
            settings_audio_output = gr.Audio(label="설정 안내", autoplay=True)
            settings_input = gr.Audio(sources=["microphone"], type="numpy")
            settings_confirmation = gr.Audio(label="설정 확인", autoplay=True)
    
    # 메인 음성 입력 처리
    main_audio_input.change(
        fn=process_voice,
        inputs=[main_audio_input, current_mode],
        outputs=[
            current_mode,
            voice_section,
            settings_section,
            settings_audio_output
        ]
    )
    
    # 설정 입력 처리
    settings_input.change(
        fn=process_settings_input,
        inputs=[settings_input, current_settings, current_mode],
        outputs=[
            current_settings,
            settings_confirmation,
            current_mode,
            settings_section,
            voice_section
        ]
    )
    
    # 음성 채팅 처리
    voice_chat_input.change(
        fn=process_voice_chat,
        inputs=[voice_chat_input, current_settings],
        outputs=voice_chat_output
    )

if __name__ == "__main__":
    demo.launch(server_port=8501, server_name='0.0.0.0')