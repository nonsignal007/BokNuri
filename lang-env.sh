
## set STT - ENV
apt-get update
apt-get install -y libsndfile1 ffmpeg libffi-dev portaudio19-dev

pip install cython
pip install -r stt-requirements.txt
pip install -r tts-requirements.txt
pip install -r requirements.txt

python -m unidic download