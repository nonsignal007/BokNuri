import gradio as gr
import speech_recognition as sr
import scipy.io.wavfile as wavfile
import tempfile
import os

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

def check_command(text):
    """명령어 확인"""
    if text is None:
        return None
    if "텍스트" in text:
        return "text"
    elif "설정" in text:
        return "settings"
    elif "음성" in text:
        return "voice"
    return None

def process_voice(audio, current_mode):
    """음성 입력 처리"""
    if audio is None:
        return current_mode, gr.update(), gr.update(), gr.update()
    
    # 음성을 텍스트로 변환
    text = speech_to_text(audio)
    
    # 명령어 확인
    command = check_command(text)
    
    if command:
        # 컴포넌트 가시성 업데이트
        voice_visible = command == "voice"
        text_visible = command == "text"
        settings_visible = command == "settings"
        
        return command, \
               gr.update(visible=voice_visible), \
               gr.update(visible=text_visible), \
               gr.update(visible=settings_visible)
    
    return current_mode, gr.update(), gr.update(), gr.update()

def process_text(text):
    """텍스트 입력 처리"""
    return f"입력하신 텍스트: {text}"

with gr.Blocks() as demo:
    # 현재 모드 상태 저장
    current_mode = gr.State("voice")
    
    # 상단 고정 음성 입력 섹션
    with gr.Row():
        gr.Markdown("# 음성 제어 AI 비서")
    with gr.Row():
        audio_input = gr.Audio(sources=["microphone"], type="numpy")
    
    # 음성 챗봇 섹션
    with gr.Row(visible=True) as voice_section:
        with gr.Column():
            gr.Markdown("## 음성 챗봇")
            voice_chat_input = gr.Audio(sources=["microphone"], type="numpy")
            voice_chat_output = gr.Audio(label="AI 응답")
    
    # 텍스트 챗봇 섹션
    with gr.Row(visible=False) as text_section:
        with gr.Column():
            gr.Markdown("## 텍스트 챗봇")
            text_input = gr.Textbox(label="메시지 입력", lines=2)
            text_response = gr.Textbox(label="AI 응답", lines=2)
    
    # 설정 섹션
    with gr.Row(visible=False) as settings_section:
        with gr.Column():
            gr.Markdown("## 설정")
            language = gr.Dropdown(
                choices=["한국어", "영어", "일본어"],
                label="언어 설정",
                value="한국어"
            )
            model = gr.Radio(
                choices=["GPT-3", "GPT-4"],
                label="AI 모델",
                value="GPT-3"
            )
    
    # 음성 입력 이벤트 처리
    audio_input.change(
        fn=process_voice,
        inputs=[audio_input, current_mode],
        outputs=[
            current_mode,
            voice_section,
            text_section,
            settings_section
        ]
    )
    
    # 텍스트 입력 이벤트 처리
    text_input.submit(
        fn=process_text,
        inputs=[text_input],
        outputs=[text_response]
    )

if __name__ == "__main__":
    demo.launch()