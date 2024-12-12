import gradio as gr
import speech_recognition as sr
import scipy.io.wavfile as wavfile
import tempfile
import os
import re
from gtts import gTTS
from pathlib import Path
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
from rank_bm25 import BM25Okapi

@dataclass
class SearchResult:
    content: str
    metadata: Dict[str, Any]
    score: float
    source: str

class RetrievalQA:
    def __init__(self, llm, embedding_model):
        self.llm = llm
        self.embedding_model = embedding_model
        self.pdf_db = None
        self.json_db = None
        self.bm25_pdf = None
        self.bm25_json = None
        self.cache_dir = './weights'
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def initialize_db(self):
        self.pdf_db = load_pdf_faiss_db(self.embedding_model)
        self.json_db = load_json_faiss_db(self.embedding_model)
        
        pdf_docs = self._get_documents(self.pdf_db)
        json_docs = self._get_documents(self.json_db)
        
        self.bm25_pdf = self._init_bm25(pdf_docs)
        self.bm25_json = self._init_bm25(json_docs)
        
    def _get_documents(self, db):
        docs = db.similarity_search("", k=1000)
        return [doc.page_content for doc in docs]
        
    def _init_bm25(self, documents):
        tokenized_docs = [doc.split() for doc in documents]
        return BM25Okapi(tokenized_docs)
    
    def get_answer(self, query: str, user_info: Dict = None) -> str:
        try:
            search_results = self.hybrid_search(query)
            context = self._build_context(search_results, user_info)
            
            prompt = f"""다음 정보를 바탕으로 질문에 답변해주세요.
            
            질문: {query}
            
            참고 정보:
            {context}
            
            위 정보를 바탕으로 친절하고 상세하게 답변해주세요."""
            
            response = self.llm.invoke(prompt)
            return response.strip()
            
        except Exception as e:
            print(f"Error in get_answer: {str(e)}")
            return "죄송합니다. 응답 생성 중 오류가 발생했습니다."

class ChatbotState:
    def __init__(self):
        self.cache_dir = './weights'
        os.makedirs(self.cache_dir, exist_ok=True)
        
        os.environ['TRANSFORMERS_CACHE'] = self.cache_dir
        os.environ['HF_HOME'] = self.cache_dir
        os.environ['HF_DATASETS_CACHE'] = os.path.join(self.cache_dir, 'datasets')
        os.environ['HUGGINGFACE_HUB_CACHE'] = self.cache_dir
        os.environ['TORCH_HOME'] = os.path.join(self.cache_dir, 'torch')
        
        try:
            print("모델 로딩 중...")
            self.llm = load_model('llama', self.cache_dir)
            
            print("임베딩 모델 로딩 중...")
            self.embedding_model = load_embedding('cuda')
            
            print("QA 시스템 초기화 중...")
            self.qa_system = RetrievalQA(self.llm, self.embedding_model)
            self.qa_system.initialize_db()
            
            self.settings = None
            print("초기화 완료!")
            
        except Exception as e:
            print(f"초기화 중 오류 발생: {str(e)}")
            raise

    def update_settings(self, settings):
        try:
            self.settings = settings
            return True
        except Exception as e:
            print(f"Settings update error: {str(e)}")
            return False

    def get_response(self, question: str) -> str:
        try:
            return self.qa_system.get_answer(question, self.settings)
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "죄송합니다. 응답 생성 중 오류가 발생했습니다."

# ChatbotState 초기화
print("챗봇 초기화 중...")
chatbot_state = ChatbotState()

def check_command(text):
    if text is None:
        return None
    elif "설정" in text:
        return "settings"
    elif "음성" in text:
        return "voice"
    return None

def process_voice(audio, current_mode):
    if audio is None:
        return current_mode, gr.update(), gr.update(), None
    
    text = speech_to_text(audio)
    command = check_command(text)
    
    if command:
        voice_visible = command == "voice"
        settings_visible = command == "settings"
        
        settings_audio = None
        if settings_visible:
            setting_text = "안녕하세요. 복지 서비스 설정 페이지 입니다. 이름, 성별, 나이, 소득분위, 장애등급을 말씀해주세요"
            settings_audio = text_to_speech(setting_text)
        
        return command, \
               gr.update(visible=voice_visible), \
               gr.update(visible=settings_visible), \
               settings_audio
    
    return current_mode, gr.update(), gr.update(), None

def process_settings_input(audio, current_settings, current_mode):
    if audio is None:
        return current_settings, None, current_mode, gr.update(), gr.update()
    
    text = speech_to_text(audio)
    if text is None:
        return current_settings, None, current_mode, gr.update(), gr.update()
    
    if any(keyword in text for keyword in ['네', '예', '맞아', '맞습니다']):
        start_message = "이제부터 복지 서비스를 시작하겠습니다. 궁금하신점이 무엇인가요?"
        start_audio = text_to_speech(start_message)
        chatbot_state.update_settings(current_settings)
        return current_settings, start_audio, "voice", gr.update(visible=False), gr.update(visible=True)
    
    income, disability, age, gender, name = extract_settings(text)
    
    new_settings = current_settings or {}
    if income: new_settings['income_level'] = income
    if disability: new_settings['disability_grade'] = disability
    if age: new_settings['age'] = age
    if gender: new_settings['gender'] = gender
    if name: new_settings['name'] = name
    
    confirmation = f"입력하신 정보를 확인해드립니다. "
    if name: confirmation += f"{name}님, "
    if income: confirmation += f"소득분위 {income}분위, "
    if disability: confirmation += f"장애 {disability}등급, "
    if age: confirmation += f"나이 {age}세, "
    if gender: confirmation += f"{gender}이시네요. "
    confirmation += "입력하신 정보가 맞다면 '네', 수정이 필요하시다면 '아니오'라고 말씀해 주세요."
    
    confirmation_audio = text_to_speech(confirmation)
    
    return new_settings, confirmation_audio, current_mode, gr.update(), gr.update()

def process_voice_chat(audio, current_settings):
    if audio is None:
        return text_to_speech("음성 입력이 없습니다.")
    
    text = speech_to_text(audio)
    if text is None:
        return text_to_speech("죄송합니다. 음성을 인식하지 못했습니다.")
    
    try:
        print(f"\n=== 새로운 질문 ===")
        print(f"인식된 텍스트: {text}")
        
        response = chatbot_state.get_response(text)
        print(f"\n생성된 응답: {response}")
        
        return text_to_speech(response)
    
    except Exception as e:
        print(f"Error in process_voice_chat: {str(e)}")
        return text_to_speech("죄송합니다. 응답 생성 중 오류가 발생했습니다.")

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
    
    main_audio_input.change(
        fn=process_voice,
        inputs=[main_audio_input, current_mode],
        outputs=[current_mode, voice_section, settings_section, settings_audio_output]
    )
    
    settings_input.change(
        fn=process_settings_input,
        inputs=[settings_input, current_settings, current_mode],
        outputs=[current_settings, settings_confirmation, current_mode, settings_section, voice_section]
    )
    
    voice_chat_input.change(
        fn=process_voice_chat,
        inputs=[voice_chat_input, current_settings],
        outputs=voice_chat_output
    )

if __name__ == "__main__":
    demo.launch(server_port=8501, server_name='0.0.0.0', share=True)