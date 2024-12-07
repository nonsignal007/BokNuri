import base64
import yaml
import os
import numpy as np
import streamlit as st
import sounddevice as sd
import simpleaudio as sa
import threading
import time

from scipy.io.wavfile import write
from src.ui_components import UIComponents
from pathlib import Path

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
        # 클래스 레벨 변수
        self._stream = None
        self._recording_thread = None
        self._is_recording = False
        self._audio_data = []

        # session_state 초기화
        if 'is_recording' not in st.session_state:
            st.session_state.is_recording = False
        if 'click_count' not in st.session_state:
            st.session_state.click_count = 0

        st.set_page_config(
            page_title="시각장애인을 위한 보이스 봇",
            page_icon="🎤",
            layout="centered"
        )

    def recording_process(self):
        """백그라운드에서 실행될 녹음 프로세스"""
        try:
            self._stream = sd.InputStream(
                samplerate=self.config.get('system_sound')['sampling_rate'],
                channels=1,
                dtype='int16'
            )
            self._stream.start()
            
            while self._is_recording:
                frame, _ = self._stream.read(1024)
                self._audio_data.append(frame)
                time.sleep(0.001)
                
        except Exception as e:
            print(f'Recording error: {e}')
            self._is_recording = False
            st.session_state.is_recording = False

    def recording(self):
        """녹음 시작"""
        self._audio_data = []
        self._is_recording = True
        
        if self._recording_thread is None or not self._recording_thread.is_alive():
            self._recording_thread = threading.Thread(target=self.recording_process)
            self._recording_thread.start()
    
    def stop_recording(self):
        """녹음 중지"""
        self._is_recording = False

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self._recording_thread and self._recording_thread.is_alive():
            self._recording_thread.join(timeout=1.0)
            self._recording_thread = None

        if len(self._audio_data) > 0:
            audio_data = np.concatenate(self._audio_data, axis=0)
            # 녹음 파일 저장
            write(
                os.path.join(
                    self.config.get('app_path'),
                    'src',
                    self.config.get('system_sound')['recorded_file_path']
                ),
                self.config.get('system_sound')['sampling_rate'],
                audio_data
            )
            self._audio_data = []
            return audio_data
        return None
        
    def handle_recording_state(self, col2):
        if not st.session_state.is_recording:
            mic_path = os.path.join(self.config.get('app_path'), 'src', 'mic_appear.gif')
            if UIComponents.create_clickable_images(mic_path, "클릭하여 녹음 시작") == 0:
                st.session_state.is_recording = True
                self.recording()
        else:
            mic_path = os.path.join(self.config.get('app_path'), 'src', 'mic_recording.gif')
            if UIComponents.create_clickable_images(mic_path, "클릭하여 녹음 종료") == 0:
                st.session_state.is_recording = False
                recorded_data = self.stop_recording()
                
    def run(self):
        col1, col2, col3 = st.columns([1,2,1])

        with col2:
            st.title("보이스 봇")
            self.handle_recording_state(col2)


if __name__ == "__main__":
    print(os.getcwd())
    VoiceBot().run()