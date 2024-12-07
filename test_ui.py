import streamlit as st
from st_clickable_images import clickable_images
from PIL import Image
import base64
from pathlib import Path

# 페이지 설정
st.set_page_config(
    page_title="시각장애인을 위한 보이스 봇",
    page_icon="🎤",
    layout="centered"
)

def listening():
    print("Listening function is running...")
    st.write("Listening function is running...")
    pass

def get_image_as_base64(file_path):
    with Path(file_path).open('rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def main():
    # 세션 스테이트 초기화
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False

    # 제목
    st.title("마이크 앱")

    # 상태 표시를 위한 컨테이너
    status_container = st.empty()

    if not st.session_state.is_recording:
        # 초기 상태: mic_appear.gif
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            base64_gif = get_image_as_base64('mic_appear.gif')
            clicked = clickable_images(
                [f"data:image/gif;base64,{base64_gif}"],
                titles=["클릭하여 녹음 시작"],
                div_style={"display": "flex", "justify-content": "center", "align-items": "center"},
                img_style={"cursor": "pointer", "transition": "transform .3s", "width": "300px"}
            )
            
            if clicked == 0:  # 이미지가 클릭되었을 때
                st.session_state.is_recording = True
                st.rerun()
        
    else:
        # 녹음 중 상태
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            # mic_recording.gif를 클릭 가능하게 표시
            base64_recording_gif = get_image_as_base64('mic_recording.gif')
            clicked = clickable_images(
                [f"data:image/gif;base64,{base64_recording_gif}"],
                titles=["클릭하여 녹음 중지"],
                div_style={"display": "flex", "justify-content": "center", "align-items": "center"},
                img_style={"cursor": "pointer", "transition": "transform .3s", "width": "300px"}
            )
            
            status_container.write("녹음 중...")
            
            # listening 함수 실행
            listening()
            
            if clicked == 0:  # 녹음 중인 마이크를 클릭했을 때
                st.session_state.is_recording = False
                st.rerun()

if __name__ == "__main__":
    main()