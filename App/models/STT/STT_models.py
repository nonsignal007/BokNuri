# Real-time Inference for Wav2Vec-Ksponspeech
# Author : Taehyoung Kim

import torch
import librosa
from models.STT.jaso import jaso
import warnings
warnings.filterwarnings('ignore')

from transformers import Wav2Vec2Processor,  Wav2Vec2ForCTC

from src.set_weights_dir import set_weights_dir



class STTModel:
    def __init__(self, config):
        self.weights_dir = set_weights_dir(config)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
        self.processor = Wav2Vec2Processor.from_pretrained(
                                            "Taeham/wav2vec2-ksponspeech",
                                            cache_dir = self.weights_dir)
        self.model = Wav2Vec2ForCTC.from_pretrained(
                                    "Taeham/wav2vec2-ksponspeech",
                                    cache_dir=self.weights_dir).to(self.device)
    
    def transcribe(self, audio_bytes):
        audio_bytes = librosa.util.buf_to_float(audio_bytes)
        tmp = self.processor(audio_bytes, sampling_rate = 16000, return_tensors='pt', padding=True)
        input_values = tmp.input_values.to(self.device)
        with torch.no_grad():
            input_values = input_values.type(torch.FloatTensor).to(self.device)
            logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        pred = self.processor.batch_decode(predicted_ids)
        return jaso().to_sentence(''.join(pred))


