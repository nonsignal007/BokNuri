
## set STT - ENV
apt-get update
apt-get install -y libsndfile1 ffmpeg libffi-dev portaudio19-dev

pip install cython
pip install -r stt-requirements.txt
pip install -r tts-requirements.txt
pip install -r requirements.txt

python -m unidic download



## set langchain - ENV

# mkdir test_App/src/weights

# pip install transformers
# pip install langchain
# pip install langchain-community
# pip install langchain-core
# pip install langchain_huggingface
# pip install pypdf
# pip install faiss-gpu
# pip install 'accelerate>=0.26.0'
# pip install tiktoken
# pip install loguru
# pip install sounddevice
# pip install simpleaudio
# pip install st_clickable_images
# pip install audio_recorder_streamlit
# pip install pyaudio
# pip install librosa
# pip install langchain_experimental
# pip install langchain_chroma
# pip install sentence-transformers
# # cd test_App/models/TTS
# # pip install -e .
# # python -m unidic download
# # cd ../../..
