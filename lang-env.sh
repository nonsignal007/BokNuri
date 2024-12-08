
mkdir src/weights

pip install transformers
pip install langchain
pip install langchain-community
pip install langchain-core
pip install langchain_huggingface
pip install pypdf
pip install faiss-gpu
pip install 'accelerate>=0.26.0'
pip install streamlit
pip install tiktoken
pip install loguru
pip install sounddevice
pip install simpleaudio
pip install st_clickable_imagespip install audio_recorder_streamlit
pip install pyaudio

git clone https://github.com/myshell-ai/MeloTTS.git test_App/models/TTS
cd test_App/models/TTS
pip install -e .
python -m unidic download
cd ../../..