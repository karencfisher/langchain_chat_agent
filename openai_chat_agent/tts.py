'''
Implements text to speech using pyttsx3
'''
import os
import pyttsx3
import json

class Text2Speech:
   def __init__(self):
      # get voice specification
      path = os.path.join('openai_chat_agent', 'voice.json')
      with open(path, 'r') as FP:
         info = json.load(FP)

      # initialize the engine
      self.engine = pyttsx3.init()
      voices = self.engine.getProperty('voices')
      voice_found = False
      for voice in voices:
         if voice.name == info['voice']:
            self.engine.setProperty('voice', voice.id)
            voice_found = True
            break
      if not voice_found:   
         raise ValueError(f'Voice {info["voice"]} is not found')
      self.engine.setProperty('rate', info['rate'])

   def speak(self, text):
      self.engine.say(text)
      self.engine.runAndWait()


def test():
   tts = Text2Speech()
   tts.speak('hello, this is a boring test')

if __name__ == '__main__':
   test()
   