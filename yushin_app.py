import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import os
import time
import simpleaudio as sa
import threading

st.title("시각장애인을 위한 챗봇 서비스")

# 녹음 설정
sampling_rate = 44100  # 44.1kHz
recorded_file_path = "recorded_audio.wav" # 사용자에게 받는 음원
processed_file_path = "processed_audio.wav" # LLM의 결과를 TTS 형태로 받은 음원
arrival_sound_path = "arrive.wav"  # "응답이 준비되었습니다."
wait_sound_path = "wait.wav"  # "잠시만 기다려 주세요."
service_info = "info.wav"  # "화면의 중간 부분을 터치해주세요."
replay_path = "replay.wav"  # '다시 재생을 원하시는 경우 화면 아래 부분을 터치해주세요'
none_path = "none.wav" # '도착한 응답이 없습니다.' 

# Session State init.
if "click_count" not in st.session_state:
    st.session_state.click_count = 0  
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False
if "audio_data" not in st.session_state:
    st.session_state.audio_data = [] 
if "waiting" not in st.session_state:
    st.session_state.waiting = False 
if "has_played_info" not in st.session_state:
    st.session_state.has_played_info = False  
if "processed_audio_played" not in st.session_state:
    st.session_state.processed_audio_played = False  
if "show_replay_button" not in st.session_state:
    st.session_state.show_replay_button = False  

# 녹음 함수
def record_audio(sampling_rate):
    st.session_state.audio_data = [] 
    stream = sd.InputStream(samplerate=sampling_rate, channels=1, dtype="int16")
    stream.start()
    # st.info("녹음 중입니다... 버튼을 다시 클릭하면 중지됩니다.")

    try:
        while st.session_state.is_recording:
            frame, _ = stream.read(1024)
            st.session_state.audio_data.append(frame)
    finally:
        stream.stop()
        stream.close()

# 오디오 재생 함수
def play_audio(audio_path):
    try:
        wave_obj = sa.WaveObject.from_wave_file(audio_path)
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception as e:
        st.error(f"오디오 재생 중 오류 발생: {e}")

# 서비스 시작 전 대기음 재생
if not st.session_state.has_played_info:
    play_audio(service_info)
    st.session_state.has_played_info = True

# 버튼 동작: 녹음 시작 >> 중지
if st.button("녹음 시작/중지"):
    st.session_state.click_count += 1

    if st.session_state.click_count == 1:
        st.session_state.is_recording = True
        # st.success("녹음이 시작되었습니다.")
        record_audio(sampling_rate)
    elif st.session_state.click_count == 2:
        st.session_state.is_recording = False
        st.session_state.click_count = 0

        if st.session_state.audio_data:
            audio_data_array = np.concatenate(st.session_state.audio_data, axis=0)
            write(recorded_file_path, sampling_rate, audio_data_array)
        else:
            st.warning("녹음 데이터가 없습니다. 다시 시도해주세요.")

# 녹음 중지 후 대기 상태
if not st.session_state.is_recording and os.path.exists(recorded_file_path) and not st.session_state.waiting:
    st.session_state.waiting = True

    def wait_and_play():
        while not os.path.exists(processed_file_path):
            play_audio(wait_sound_path)
            time.sleep(2)

        if os.path.exists(processed_file_path):
            play_audio(arrival_sound_path)
            play_audio(processed_file_path)
            st.session_state.processed_audio_played = True
            st.session_state.show_replay_button = True

        st.session_state.waiting = False

    threading.Thread(target=wait_and_play, daemon=True).start()

# "다시 실행" 버튼 클릭 시 처리
if st.button("다시 실행"):
    if os.path.exists(processed_file_path):
        play_audio(processed_file_path)
    else:
        # st.warning("도착한 내용이 없습니다.")
        play_audio(none_path)

# 대기 상태로 전환
if not st.session_state.is_recording and os.path.exists(recorded_file_path) and not st.session_state.waiting:
    st.session_state.waiting = True

    def wait_and_play():
        while not os.path.exists(processed_file_path):
            play_audio(wait_sound_path)
            time.sleep(2)

        if os.path.exists(processed_file_path):
            play_audio(arrival_sound_path)  # 알림음 재생

        st.session_state.waiting = False

    threading.Thread(target=wait_and_play, daemon=True).start()