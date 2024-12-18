
## set STT - ENV
apt-get update
apt-get install -y libsndfile1 ffmpeg libffi-dev portaudio19-dev

pip install cython
pip install -r stt-requirements.txt
pip install -r tts-requirements.txt
pip install -r requirements.txt
pip install transformers==4.41.2 huggingface-hub==0.23.0
pip install gtts
pip install rank_bm25
# pip install 'transformers<4.27.0'

python -m unidic download