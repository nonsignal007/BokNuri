from melo.api import TTS

class TTSModel:
    def __init__(self, config):
        self.tts = TTS(language='KR', device='cuda')

    def synthesize(self, text):
        speaker_ids = self.tts.hps.data.spk2id
        output_path = 'kr.wav'
        self.tts.tts_to_file(text, speaker_ids['KR'], output_path)