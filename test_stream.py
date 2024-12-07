import streamlit as st
import pandas as pd
import numpy as np

# Streamlit 포트 설정
import socket
from streamlit.web.server import Server
from streamlit.web.server.server import PORT_NUMBER

# 포트 8501로 설정
if __name__ == '__main__':
    PORT_NUMBER = 8501

# 페이지 제목 설정
st.title('내 첫 번째 Streamlit 앱')

# 텍스트 입력 위젯
user_input = st.text_input('이름을 입력하세요', '게스트')
st.write(f'안녕하세요, {user_input}님!')

# 슬라이더 위젯
number = st.slider('숫자를 선택하세요', 0, 100, 50)
st.write(f'선택한 숫자: {number}')

# 차트 그리기
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['데이터1', '데이터2', '데이터3'])

st.line_chart(chart_data)

# 사이드바 추가
st.sidebar.header('사이드바 메뉴')
option = st.sidebar.selectbox(
    '원하는 옵션을 선택하세요',
    ['옵션 1', '옵션 2', '옵션 3']
)
st.sidebar.write('선택한 옵션:', option)

# 현재 실행 중인 포트 표시
st.sidebar.write(f'현재 포트: {PORT_NUMBER}')