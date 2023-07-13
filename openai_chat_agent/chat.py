import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv
from langchain.chat_models.openai import ChatOpenAI

from langchain.utilities.google_search import GoogleSearchAPIWrapper
from langchain.agents import (initialize_agent,
                              Tool,
                              AgentType)
from langchain.tools import tool
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS
import faiss

try:
    from openai_chat_agent.vectorstore import VectorStoreRetrieverMemory
except:
    from vectorstore import VectorStoreRetrieverMemory


class ChatAgent:
    debug = False

    def __init__(self, verbose=True):
        load_dotenv()

        # setup logging
        now = datetime.now()
        logfile = f'chatlog-{now.strftime("%m.%d.%Y-%H.%M.%S")}.log'
        logpath = os.path.join('openai_chat_agent', 'logs', logfile)
        logging.basicConfig(handlers=[logging.FileHandler(logpath, 'w', 'utf-8')], 
                            level=logging.INFO, 
                            format='%(message)s')
        self.logger = logging.getLogger()

        # setup chat model
        chat = ChatOpenAI(model='gpt-4', temperature=0, verbose=verbose)

        embeddings = OpenAIEmbeddings()
        embedding_fn = embeddings.embed_query
        embedding_size = 1536
        index = faiss.IndexFlatL2(embedding_size)

        if  os.path.exists('conversation.db'):
            vectorstore = FAISS.load_local('conversation.db', embeddings)
        else:
            vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})

        retriever = vectorstore.as_retriever(search_kwargs=dict(k=3))
        memory = VectorStoreRetrieverMemory(retriever=retriever, 
                                            memory_key='chat_history',
                                            return_messages=True)

        # setup tools and agent
        search=GoogleSearchAPIWrapper()
        tools = [Tool(name='Search',
                    func=search.run,
                    description="Useful if you are asked about current events or current information on topics.")
                ]

        self.conversation = initialize_agent(
            tools=tools,
            llm=chat, 
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            verbose=verbose
        )

        # setup sys prompt
        profile_path = os.path.join('openai_chat_agent', 'sys_prompt.txt')
        with open(profile_path, 'r') as FILE:
           sys_prompt = FILE.read()

        prompt = self.conversation.agent.create_prompt(
            system_message=sys_prompt,
            tools=tools
        )
        self.conversation.agent.llm_chain.prompt=prompt

    def __call__(self, text):
        self.logger.info(f'Human: {text}\n')
        try:
            result = self.conversation.run(input=text)
            self.logger.info(f'AI: {result}\n')
            return None, result
        except Exception as ERR:
            self.logger.error(f'ERROR {ERR}\n')
            if self.debug:
                raise ERR
            return ERR, None    

    
class Bot:
    def __init__(self, voice=False, verbose=True):
        self.recog = SpeechRecognize()
        self.chat_agent = ChatAgent(verbose=verbose)

        if voice:
            print('Initializing speech...')
            try:
                from openai_chat_agent.vosk_recognizer import SpeechRecognize
                from openai_chat_agent.tts import Text2Speech
            except:
                from vosk_recognizer import SpeechRecognize
                from tts import Text2Speech

            self.tts = Text2Speech()
            self.voice = voice
            print('Done.')

    def loop(self):
        # start loop
        text = 'hello'
        while True:
            err, result = self.chat_agent(text)
            if err is None:
                if self.voice:
                    print('\rtalking...     ', end='')
                    self.tts.speak(result)
                else:
                    print(f'AI: {result}\n')
            else:
                print(f'\rError: {err}')

            if text.lower() == 'goodbye':
                break

            if self.voice:
                print('\rlistening...     ', end='')
                text = self.recog.speech_to_text()
            else:
                text = input('Human: ')
                
        print('\ndone!')


def main():
    params = {'voice': False, 'verbose': False}
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg in params.keys():
                params[arg] = True
            else:
                raise ValueError('Invalid command line argument {sys.argv[1]}')
    bot = Bot(voice=params['voice'], verbose=params['verbose'])
    bot.loop()

if __name__ == '__main__':
    main()
