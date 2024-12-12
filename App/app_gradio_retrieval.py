import gradio as gr
import speech_recognition as sr
import scipy.io.wavfile as wavfile
import tempfile
import os
import time
import re
from gtts import gTTS
from pathlib import Path
import numpy as np
from typing import Dict, List, Any, Optional

# STT
from models.STT.KoreanStreamingASR.src.utils.transcriber import DenoiseTranscriber
from models.STT.KoreanStreamingASR.src.utils.argparse_config import setup_arg_parser

# RAG
from models.RAG.load_embedding import create_embedding_model, create_dbs, create_law_db , create_qa_db
from models.RAG.load_llm import load_model
from models.RAG.Langchain_retrieval import LegalQASystem

from langchain.vectorstores import FAISS

########## STT, TTS Functions ###############
def speech_to_text(audio):
    """음성을 텍스트로 변환"""
    if audio is None:
        return None
    
    sample_rate, data = audio
    
    try:
        # 임시 wav 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            wavfile.write(temp_audio.name, sample_rate, data)
            
            # DenoiseTranscriber 설정
            parser = setup_arg_parser()
            args = parser.parse_args()
            args.mode = 'file'
            args.inference = False
            args.audio_path = temp_audio.name
            
            # STT 모델로 변환
            transcriber = DenoiseTranscriber(args)
            transcription = transcriber.transcribe(audio_path=temp_audio.name)
            
            if transcription is not None:
                return str(transcription)
            return None
            
    except Exception as e:
        print(f"STT 에러: {str(e)}")
        return None
    finally:
        try:
            os.unlink(temp_audio.name)
        except:
            pass

def text_to_speech(text):
    """텍스트를 음성으로 변환"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            tts = gTTS(text=text, lang='ko')
            tts.save(fp.name)
            return fp.name
    except Exception as e:
        return f"음성 변환 오류: {str(e)}"

########## Command Processing ###############
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
        
        settings_audio = None
        if settings_visible:
            setting_text = "안녕하세요. 복지 서비스 설정 페이지입니다. 이름, 성별, 나이, 소득분위, 장애등급을 말씀해주세요."
            settings_audio = text_to_speech(setting_text)
        
        return command, gr.update(visible=voice_visible), gr.update(visible=settings_visible), settings_audio
    
    return current_mode, gr.update(), gr.update(), None

########## Settings Processing ###############
def extract_settings(text):
    """사용자 음성에서 설정 정보 추출"""
    if text is None or not isinstance(text, str):
        print(f"텍스트 타입 에러: {type(text)}")
        return None, None, None, None, None
    
    try:
        # 소득분위 추출
        income_patterns = [
            r'(\d+)\s*분위',
            r'(\d+)분위',
            r'소득\s*(\d+)\s*분위'
        ]
        income_level = None
        for pattern in income_patterns:
            income_match = re.search(pattern, text)
            if income_match:
                income_level = income_match.group(1)
                break
                
        # 장애등급 추출
        disability_patterns = [
            r'(\d+)\s*(?:급|등급)',
            r'장애\s*(\d+)\s*급',
            r'장애\s*(\d+)\s*등급'
        ]
        disability_grade = None
        for pattern in disability_patterns:
            disability_match = re.search(pattern, text)
            if disability_match:
                disability_grade = disability_match.group(1)
                break
        
        # 나이 추출
        age_patterns = [
            r'(\d+)\s*(?:살|세)',
            r'나이\s*(\d+)',
            r'(?:스물|서른|마흔|쉰|예순|일흔|여든|아흔)\s*(?:한|두|세|네|다섯|여섯|일곱|여덟|아홉|열)\s*살',
            r'스물\s*살',
            r'서른\s*살',
            r'마흔\s*살'
        ]
        
        age = None
        for pattern in age_patterns:
            age_match = re.search(pattern, text)
            if age_match:
                age_text = age_match.group(0)
                # 한글 숫자를 아라비아 숫자로 변환
                korean_numbers = {
                    '스물': 20, '서른': 30, '마흔': 40, '쉰': 50, '예순': 60,
                    '일흔': 70, '여든': 80, '아흔': 90,
                    '한': 1, '두': 2, '세': 3, '네': 4, '다섯': 5,
                    '여섯': 6, '일곱': 7, '여덟': 8, '아홉': 9, '열': 10
                }
                
                if any(num in age_text for num in korean_numbers.keys()):
                    age_num = 0
                    for k, v in korean_numbers.items():
                        if k in age_text:
                            if v >= 20:  # 십단위
                                age_num = v
                            else:  # 일단위
                                age_num += v
                    age = str(age_num)
                else:
                    age = age_match.group(1)
                break
        
        # 성별 추출
        gender = None
        gender_patterns = {
            '남성': [r'남자', r'남성', r'남'],
            '여성': [r'여자', r'여성', r'여']
        }
        
        for gender_type, patterns in gender_patterns.items():
            if any(re.search(pattern, text) for pattern in patterns):
                gender = gender_type
                break
        
        # 이름 추출
        name_patterns = [
            r'(?:저는|나는|전|난)?\s*([가-힣]{2,4})[이가](?:이고|고|예요|에요|입니다|이야|야)',
            r'이름[은는이가]?\s*([가-힣]{2,4})',
            r'([가-힣]{2,4})(?:이고|고|예요|에요|입니다|이야|야)',
        ]
        
        name = None
        for pattern in name_patterns:
            name_match = re.search(pattern, text)
            if name_match:
                name = name_match.group(1)
                break
                
        print(f"추출된 정보: 이름({name}), 나이({age}), 성별({gender}), 소득분위({income_level}), 장애등급({disability_grade})")
        
        return income_level, disability_grade, age, gender, name
        
    except Exception as e:
        print(f"설정 추출 에러: {str(e)}")
        return None, None, None, None, None

def process_settings_input(audio, current_settings, current_mode):
    """설정 음성 입력 처리"""
    if audio is None:
        return current_settings, None, current_mode, gr.update(), gr.update()
    
    text = speech_to_text(audio)
    if text is None:
        return current_settings, None, current_mode, gr.update(), gr.update()
    
    # 확인 응답 체크
    if any(keyword in text for keyword in ['네', '예', '맞아', '맞습니다']):
        start_message = "이제부터 복지 서비스를 시작하겠습니다. 궁금하신 점이 무엇인가요?"
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

########## Voice Chat Processing ###############
def process_voice_chat(audio, current_settings):
    """음성 채팅 처리"""
    if audio is None:
        return text_to_speech("음성 입력이 없습니다.")
    
    text = speech_to_text(audio)
    if text is None:
        return text_to_speech("죄송합니다. 음성을 인식하지 못했습니다.")
    
    try:
        print(f"\n=== 새로운 질문 ===")
        print(f"인식된 텍스트: {text}")
        
        # 설정이 변경되었다면 상태 업데이트
        if current_settings != chatbot_state.settings:
            chatbot_state.update_settings(current_settings)
            print(f"사용자 설정: {format_user_info(current_settings)}")
        
        # 응답 생성
        response = chatbot_state.get_response(text)
        print(f"\n생성된 응답: {response}")
        
        # 응답을 음성으로 변환
        return text_to_speech(response)
    
    except Exception as e:
        print(f"Error in process_voice_chat: {str(e)}")
        return text_to_speech("죄송합니다. 응답 생성 중 오류가 발생했습니다.")

########## Gradio Interface ###############
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
        """챗봇 상태 및 LegalQASystem 초기화"""
        # 캐시 디렉토리 설정
        self.cache_dir = './weights'
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # DB 경로 설정
        self.law_db_path = "/workspace/LangEyE/App/data/laws"
        self.qa_db_path = "/workspace/LangEyE/App/data/qa"
        
        # 환경 변수 설정
        self._setup_environment()
        
        try:
            # 모델 및 임베딩 초기화
            print("모델 로딩 중...")
            self.llm = load_model('llama', self.cache_dir)
            
            print("임베딩 모델 로딩 중...")
            self.embedding_model = create_embedding_model()
            
            # FAISS Vector DB 초기화
            print("FAISS Vector DB 초기화 중...")
            self._initialize_vector_dbs()
            
            # 하이브리드 검색을 위한 초기 데이터 준비
            print("하이브리드 검색 초기화 중...")
            self.law_documents = self._prepare_law_documents()
            self.qa_documents = self._prepare_qa_documents()
            
            # QA 시스템 초기화
            print("QA 시스템 초기화 중...")
            self._initialize_qa_system()
            
            self.settings = None
            print("초기화 완료!")
            
        except Exception as e:
            print(f"초기화 중 오류 발생: {str(e)}")
            raise
    
    def _setup_environment(self):
        """환경 변수 설정"""
        os.environ['TRANSFORMERS_CACHE'] = self.cache_dir
        os.environ['HF_HOME'] = self.cache_dir
        os.environ['HF_DATASETS_CACHE'] = os.path.join(self.cache_dir, 'datasets')
        os.environ['HUGGINGFACE_HUB_CACHE'] = self.cache_dir
        os.environ['TORCH_HOME'] = os.path.join(self.cache_dir, 'torch')
    
    def _initialize_vector_dbs(self):
        """Vector DB 초기화 또는 생성"""
        # 법령 DB
        if os.path.exists(self.law_db_path):
            print("기존 법령 DB 로딩 중...")
            self.law_db = FAISS.load_local(self.law_db_path, self.embedding_model, allow_dangerous_deserialization=True)
        else:
            print("법령 DB 생성 중...")
            self.law_db = create_law_db(self.embedding_model, self.law_db_path)
        
        # QA DB
        if os.path.exists(self.qa_db_path):
            print("기존 QA DB 로딩 중...")
            self.qa_db = FAISS.load_local(self.qa_db_path, self.embedding_model, allow_dangerous_deserialization=True)
        else:
            print("QA DB 생성 중...")
            self.qa_db = create_qa_db(self.embedding_model, self.qa_db_path)
        
        if not all([self.law_db, self.qa_db]):
            raise Exception("하나 이상의 DB 초기화 실패")
    
    def _initialize_qa_system(self):
        """QA 시스템 초기화"""
        self.qa_system = LegalQASystem(
            custom_llm=self.llm,
            custom_embeddings=self.embedding_model,
            law_db=self.law_db,
            qa_db=self.qa_db
        )
        
        # BM25 초기화
        self.qa_system.search_engine.initialize_bm25(
            self.law_documents,
            self.qa_documents
        )
    
    def _prepare_law_documents(self) -> List[Dict]:
        """법령 문서 준비"""
        try:
            docs = self.law_db.similarity_search("", k=1000)
            return [
                {
                    "content": doc.page_content,
                    "metadata": {
                        **doc.metadata,
                        "doc_type": "law"
                    }
                }
                for doc in docs
            ]
        except Exception as e:
            print(f"법령 문서 준비 중 오류: {str(e)}")
            return []
    
    def _prepare_qa_documents(self) -> List[Dict]:
        """QA 문서 준비"""
        try:
            docs = self.qa_db.similarity_search("", k=1000)
            return [
                {
                    "content": doc.page_content,
                    "metadata": {
                        **doc.metadata,
                        "doc_type": "qa"
                    }
                }
                for doc in docs
            ]
        except Exception as e:
            print(f"QA 문서 준비 중 오류: {str(e)}")
            return []

    def get_response(self, question: str) -> str:
        """질문에 대한 응답 생성"""
        try:
            response = self.qa_system.answer_question(question, self.settings)
            self._log_search_results(response)
            # '답변:' 부분만 포함하도록 수정
            answer = response['answer']
            if ':' in answer:
                answer = answer.split(':', 1)[1].strip()
            return f"답변: {answer}"
                
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "죄송합니다. 응답 생성 중 오류가 발생했습니다."
    
    def _log_search_results(self, response):
        """검색 결과 로깅"""
        print(f"\n=== 검색 결과 ===")
        if 'search_results' in response:
            print("\n검색된 문서:")
            for i, result in enumerate(response['search_results'], 1):
                print(f"\n[문서 {i}]")
                print(f"- 출처: {result.get('source', '불명')}")
                
                if result.get('source') == 'law':
                    print(f"- 법령명: {result['metadata'].get('law_title', '')}")
                    print(f"- 조항: {result['metadata'].get('article_number', '')}")
                    print(f"- 제목: {result['metadata'].get('article_subject', '')}")
                else:  # qa
                    print(f"- 질문: {result['metadata'].get('question', '')}")
                
                print(f"- 점수: {result.get('score', 0):.3f}")

    def update_settings(self, settings):
        """사용자 설정 업데이트"""
        try:
            self.settings = settings
            return True
        except Exception as e:
            print(f"Settings update error: {str(e)}")
            return False

print("챗봇 초기화 중...")
chatbot_state = ChatbotState()

with gr.Blocks() as demo:
    current_mode = gr.State("voice")
    current_settings = gr.State(None)
    
    welcome_text = "안녕하세요. 서울특별시 구로구의 시각장애인들을 위한 보이스 복지 서비스입니다. 설정하신 정보에 맞게 복지 정보를 알려드릴게요. 만약 설정하신게 없으시다면 '설정'을 말씀해 주세요."
    welcome_audio = text_to_speech(welcome_text)
    gr.Audio(welcome_audio, autoplay=True, visible=False)
    
    with gr.Row():
        gr.Markdown("# 복지 Q&A Voice Bot")
    
    with gr.Row():
        main_audio_input = gr.Audio(sources=["microphone"], type="numpy", label="음성 명령")
    
    with gr.Row(visible=True) as voice_section:
        with gr.Column():
            gr.Markdown("## 복지 서비스")
            voice_chat_input = gr.Audio(sources=["microphone"], type="numpy", label="질문하기")
            voice_chat_output = gr.Audio(label="AI 응답")
    
    with gr.Row(visible=False) as settings_section:
        with gr.Column():
            gr.Markdown("## 설정")
            settings_audio_output = gr.Audio(label="설정 안내", autoplay=True)
            settings_input = gr.Audio(sources=["microphone"], type="numpy", label="설정하기")
            settings_confirmation = gr.Audio(label="설정 확인", autoplay=True)
    
    # 이벤트 핸들러
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