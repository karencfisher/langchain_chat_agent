'''
Implements Vosk speech recognition
'''

import json
import os
import vosk
import pyaudio
import numpy as np

class SpeechRecognize:
    def __init__(self):
        path = os.path.join('openai_chat_agent', 'vosk_config.json')
        with open(path, 'r') as FP:
            self.config = json.load(FP)
        vosk.SetLogLevel(-1)
        model_path = os.path.join('openai_chat_agent', self.config['model'])
        model = vosk.Model(model_path)
        self.recognizer = vosk.KaldiRecognizer(model, 16000)

    def speech_to_text(self):
        mic = pyaudio.PyAudio()
        stream = mic.open(format=pyaudio.paInt16,
                        channels = self.config['channels'],
                        rate=self.config['rate'],
                        input=True,
                        frames_per_buffer=self.config['chunk'] * 2)
        stream.start_stream()

        while True:
            data = stream.read(self.config['chunk'])
            if self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                break
        return result['text']
        

def test():
    sr = SpeechRecognize()
    text = sr.speech_to_text()
    print(text)

if __name__ == '__main__':
    test()
