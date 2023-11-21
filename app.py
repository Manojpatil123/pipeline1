from PyPDF2 import PdfReader
import uvicorn
from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse
from queue import Queue
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Milvus
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import Replicate
from threading import Thread
import os
from threading import Thread
from queue import Queue, Empty
from threading import Thread
from collections.abc import Generator
from langchain.callbacks.base import BaseCallbackHandler
from typing import  Any
from langchain.tools import DuckDuckGoSearchRun
from langchain.vectorstores import Milvus
from langchain.tools import DuckDuckGoSearchRun
import requests
from flask import Flask,jsonify,request,send_file,render_template,Response,stream_with_context
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chains import SequentialChain
import whisper
from pydub import AudioSegment
import librosa
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import torch
from transformers import pipeline
import textwrap
from io import BytesIO
import requests
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from PIL import Image
import io
import numpy as np
from langchain.utilities import SerpAPIWrapper
from faster_whisper import WhisperModel
import tiktoken
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import runpod
from langchain.llms import HuggingFaceTextGenInference


from flask import Flask,jsonify,request,send_file,render_template,Response,stream_with_context
from PyPDF2 import PdfReader
import os
import glob
import spacy
from transformers import pipeline
import numpy as np
# Library to import pre-trained model for sentence embeddings
from sentence_transformers import SentenceTransformer
# Calculate similarities between sentences
from sklearn.metrics.pairwise import cosine_similarity
# Visualization library
# package for finding local minimas
from scipy.signal import argrelextrema
import math
import pysbd
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Milvus
import torch


runpod.api_key = os.getenv("RUNPOD_API_KEY", "your_runpod_api_key")

BASE_DIR = os.getcwd()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

#image chat
disable_torch_init()

MODEL = "4bit/llava-v1.5-13b-3GB"
model_name = get_model_name_from_path(MODEL)
model_name

#load llava model
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=MODEL, model_base=None, model_name=model_name, load_4bit=True
)


def load_image(image_file):
    try:
      if image_file.startswith("http://") or image_file.startswith("https://"):
          response = requests.get(image_file)
          image = Image.open(BytesIO(response.content)).convert("RGB")
      else:
          image = Image.open(image_file).convert("RGB")
    except:
          image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
    return image


def process_image(image):
    args = {"image_aspect_ratio": "pad"}
    image_tensor = process_images([image], image_processor, args)
    return image_tensor.to(model.device, dtype=torch.float16)


CONV_MODE = "llava_v0"

def create_prompt(prompt: str):
    conv = conv_templates[CONV_MODE].copy()
    conv.system="you are sales person, you have sounds like sales person and keeping the tone of the conversation professional. you have to understand user question properly and provide answer to them.Please ensure that your responses are socially unbiased and positive in nature."
    roles = conv.roles
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    conv.append_message(roles[0], prompt)
    conv.append_message(roles[1], None)
    return conv.get_prompt(), conv

def ask_image(image: Image, prompt: str):
    image_tensor = process_image(image)
    prompt, conv = create_prompt(prompt)
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria(
        keywords=[stop_str], tokenizer=tokenizer, input_ids=input_ids
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.01,
            max_new_tokens=512,
            use_cache=True,

            stopping_criteria=[stopping_criteria],
        )
    return tokenizer.decode(
        output_ids[0, input_ids.shape[1] :], skip_special_tokens=True
    ).strip()

#intialize web search wrapper
search = DuckDuckGoSearchRun()
search1 = SerpAPIWrapper(serpapi_api_key='6b15f6f86cb16bd8bda90a913dd97f573c56d5c82ab3a0f8e41366b1a63a8f0b')
model_kwargs = {'device': 'cuda'}
#intialize emebding model
embeddings1 = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2',model_kwargs=model_kwargs)

#speach to text model
model_id = "openai/whisper-large-v3"
# Run on GPU with FP16
model_whisper = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model_whisper.to(device)
processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model_whisper,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=15,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

device = 0 if torch.cuda.is_available() else -1 
#emotion text
emotion_model = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa',device=device)

#emotion detection audio
def map_to_array(file):
        example={}
        speech, _ = librosa.load(file, sr=16000, mono=True)
        example["speech"] = speech
        return example
model_audio_emotion = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-er")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-er")

#milvus database connection
collection_name = 'LangChainCollection'
connection_args={"uri": "https://in03-6bcf3c6994e6be0.api.gcp-us-west1.zillizcloud.com",'token':'96a0af8f0d052713f9ebb52f44660ad8661331a26e30e9371e821522b717141f3e3a371f3158bdbcef3516e23f3d27307ce615d5'}
vectorstore = Milvus(connection_args=connection_args, collection_name=collection_name,embedding_function=embeddings1)

#downloading the model
'''
url = "https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf"
output_file = "llama-2-7b-chat.Q5_K_M.gguf"  # The filename you want to save the downloaded file as

if os.path.exists(output_file):
   pass
else:
  response = requests.get(url)

  if response.status_code == 200:
      with open(output_file, "wb") as file:
          file.write(response.content)
      print(f"File downloaded as {output_file}")
  else:
      print("Failed to download the file.")
'''
inference_server_url = f'https://{["id"]}-80.proxy.runpod.net'
llm = HuggingFaceTextGenInference(
    inference_server_url=inference_server_url,
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.1,
    repetition_penalty=1.03,
    streaming = True,
    max_context_length=100
)
# Defined a QueueCallback, which takes as a Queue object during initialization. Each new token is pushed to the queue.
class QueueCallback(BaseCallbackHandler):
    """Callback handler for streaming LLM responses to a queue."""

    def __init__(self, q):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.q.put(token)

    def on_llm_end(self, *args, **kwargs: Any) -> None:
        return self.q.empty()


'''
#intialize replicate llm
llm = LlamaCpp(
        model_path="llama-2-7b-chat.Q5_K_M.gguf",    #  model path
        verbose=True,
        n_gpu_layers=40,
        n_batch=512,
        n_ctx=4000,
        streaming=True,
        max_tokens=1024

    )'''



def websearch(query):
  try:
    output=search1.run(query)
  except:
      output=search.run(query)
  return output


splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings1)
relevant_filter = EmbeddingsFilter(embeddings=embeddings1, similarity_threshold=0.45)
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[splitter, redundant_filter, relevant_filter]
)

def vectorsearch(query):
    try:
        compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=vectorstore.as_retriever())

        compressed_docs = compression_retriever.get_relevant_documents(query)
        output=''
        for i in compressed_docs:
            output+=str(i.page_content.replace('\n',''))
        if len(output)<1:
           output="there is no information in context"
    except:
        output="there is no information in context"
    return output



class ThreadWithReturnValue(Thread):
    def __init__(self, group = None, target=None, name= None, args = (), kwargs = {},Verbose=None):
        Thread.__init__(self,group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None :
            self._return = self._target(*self._args,**self._kwargs)

    def join(self,*args):
        Thread.join(self,*args)
        return self._return

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT_replicate = """you are sales person, you have sounds like sales person and keeping the tone of the conversation professional. you have to understand user question properly and provide answer to them.
if you not get any infromation from context . you have to ask fallow up question to user to get more infromation about question.Please ensure that your responses are socially unbiased and positive in nature.

"""
def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT_replicate ):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template




app = Flask(__name__)


# Create a function that will return our generator
def stream(input_text,prompt,context,emotion) -> Generator:

    # Create a Queue
    q = Queue()
    job_done = object()

    # Initialize the LLM we'll be using

    llm.callbacks=[QueueCallback(q)]

    llm_chain = LLMChain(llm=llm, prompt=prompt,verbose=True,output_key="businessknowledge")
    # Create a funciton to call - this will run in a thread
    def task():
        #resp = llm(input_text)
        resp=llm_chain.run({"context": context, "question": input_text,"emotion":emotion})
        q.put(job_done)


    # Create a thread and start the function
    t = Thread(target=task)
    t.start()

    content = ""

    # Get each new token from the queue and yield for our generator
    while True:
        try:
            next_token = q.get(True, timeout=1)
            if next_token is job_done:
                break
            content += next_token
            yield next_token
        except Empty:
            continue

def stream_summarize(input_text,prompt) -> Generator:

    # Create a Queue
    q = Queue()
    job_done = object()

    # Initialize the LLM we'll be using

    llm.callbacks=[QueueCallback(q)]

    llm_chain = LLMChain(llm=llm, prompt=prompt,verbose=True,output_key="businessknowledge")
    # Create a funciton to call - this will run in a thread
    def task():
        #resp = llm(input_text)
        resp=llm_chain.run({"context": input_text})
        q.put(job_done)


    # Create a thread and start the function
    t = Thread(target=task)
    t.start()

    content = ""

    # Get each new token from the queue and yield for our generator
    while True:
        try:
            next_token = q.get(True, timeout=1)
            if next_token is job_done:
                break
            content += next_token
            yield next_token
        except Empty:
            continue

@app.get("/")
def welcome():
    """Welcome to pipeline 1"""
    return {"status": "Welcome to pipeline 1"}

@app.route('/chat',methods=['GET'])
def chat():
    query = request.args.get('query')
    emotion_labels = emotion_model(str(query))
    labels = emotion_labels[0]['label']
    emotion=labels
    docs = vectorsearch(query)
    if docs!='there is no information in context':
       # Prompt
        template = """Use the following pieces of  knowledge to answer the question . user question and knowledge will have tagged with emotions square brackets.
        you have understand user emotion and knowledge emotion then provide answer based on user emotion.
.       If you not find any relevant infromation from  knowledge .you have to ask fallow up question to user for getting more context.
        keep the answer as concise as possible.Please ensure that your responses are socially unbiased and positive in nature.don't mention your emotion tags in final answer.


        Question: {question}

        emotion: {emotion}

        knowledge: {context}

        Answer:"""
        template_replicate = get_prompt(template,DEFAULT_SYSTEM_PROMPT_replicate)
        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context" ,"question","emotion"],
            template=template_replicate,
        )
        return Response( stream(query,QA_CHAIN_PROMPT,docs,emotion), mimetype= 'text/event-stream' )

    else:
        DEFAULT_SYSTEM_PROMPT_LLM = """
        """
        template = """
        You are sales person , you have to sound like sales person.your answer sounds like sales person giving answer.
        if user query required real time data to answer give ouput as i dont have access of real time data .
        user query tagged with emotion you have to understand user emotion then provide answer based on user emotion.
        you have to think and provide answer with user emotion .keep the answer as concise as possible.Please ensure that your responses are socially unbiased and positive in nature.don't mention your emotion tags in final answer.

        query: {question}

        emotion: {emotion}

        answer:
        """
        template_replicate = get_prompt(template,DEFAULT_SYSTEM_PROMPT_LLM)
        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["question","emotion"],
            template=template_replicate,
        )
        llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT,verbose=True,output_key="businessknowledge")

        #resp = llm(input_text)
        result=llm_chain.run({"emotion":emotion, "question": query})
        list_sentence=['i apologize','apologize','i am sorry',"i don't have access","i don't have","i don't","real-time","real time",'real_time', 'realtime',"i don't have access to real-time","i apologize, but i don't have access to real-time "]

        found_sentence = next((sentence for sentence in list_sentence if sentence.lower() in result.lower()), None)
        if found_sentence:
            def stream1():
              words = result.split()
              for word in words:
                  yield word + ' '
            return Response( stream1(), mimetype= 'text/event-stream' )
        else:
            template = """Use the following pieces of context to answer the question . user question and knowledge will have tagged with emotions in square brackets.
            you have understand user emotion and knowledge emotion then provide answer based on user emotion.
            you have to summarise the context remove the unclear context first then answer user query. you need to provide correct answer. you can use your knowledge also for providing answer
            you have to think and provide answer .keep the answer as concise as possible.Please ensure that your responses are socially unbiased and positive in nature.don't mention your emotion tags in final answer.

            Question: {question}

            emotion: {emotion}

            context: {context}
            Answer:"""
            template_replicate = get_prompt(template,DEFAULT_SYSTEM_PROMPT_replicate)
            QA_CHAIN_PROMPT = PromptTemplate(
                input_variables=["context", "question","emotion"],
                template=template_replicate,
            )
            docs=search.run(query)


            return Response( stream(query,QA_CHAIN_PROMPT,docs,emotion), mimetype= 'text/event-stream' )

@app.route('/audio',methods=['POST'])
def audio():
    audio_file = request.files['audio']
    # Read the audio file using pydub
    #audio = AudioSegment.from_file(audio_file)

    #query = model.transcribe(audio)
    result = pipe(audio_file)
    #print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    query = result['text']
    dataset = map_to_array(audio_file)
    inputs = feature_extractor(dataset["speech"], sampling_rate=16000, padding=True, return_tensors="pt")

    logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    labels = [model.config.id2label[_id] for _id in predicted_ids.tolist()]

    emotion=labels
    docs = vectorsearch(query)
    if docs!='there is no information in context':
       # Prompt
        template = """Use the following pieces of  knowledge to answer the question . user question and knowledge will have tagged with emotions square brackets.
        you have understand user emotion and knowledge emotion then provide answer based on user emotion.
.       If you not find any relevant infromation from  knowledge .you have to ask fallow up question to user for getting more context.
        keep the answer as concise as possible.Please ensure that your responses are socially unbiased and positive in nature.don't mention your emotion tags in final answer.


        Question: {question}

        emotion: {emotion}

        knowledge: {context}

        Answer:"""
        template_replicate = get_prompt(template,DEFAULT_SYSTEM_PROMPT_replicate)
        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context" ,"question","emotion"],
            template=template_replicate,
        )
        return Response( stream(query,QA_CHAIN_PROMPT,docs,emotion), mimetype= 'text/event-stream' )

    else:
        DEFAULT_SYSTEM_PROMPT_LLM = """
        """
        template = """
        You are sales person , you have to sound like sales person.your answer sounds like sales person giving answer.
        if user query required real time data to answer give ouput as i dont have access of real time data .
        user query tagged with emotion you have to understand user emotion then provide answer based on user emotion.
        you have to think and provide answer with user emotion .keep the answer as concise as possible.Please ensure that your responses are socially unbiased and positive in nature.don't mention your emotion tags in final answer.

        query: {question}

        emotion: {emotion}

        answer:
        """
        template_replicate = get_prompt(template,DEFAULT_SYSTEM_PROMPT_LLM)
        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["question","emotion"],
            template=template_replicate,
        )
        llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT,verbose=True,output_key="businessknowledge")

        #resp = llm(input_text)
        result=llm_chain.run({"emotion":emotion, "question": query})
        list_sentence=['i apologize','apologize','i am sorry',"i don't have access","i don't have","i don't","real-time","real time",'real_time', 'realtime',"i don't have access to real-time","i apologize, but i don't have access to real-time "]

        found_sentence = next((sentence for sentence in list_sentence if sentence.lower() in result.lower()), None)
        if found_sentence:
            def stream1():
              words = result.split()
              for word in words:
                  yield word + ' '
            return Response( stream1(), mimetype= 'text/event-stream' )
        else:
            template = """Use the following pieces of context to answer the question . user question and knowledge will have tagged with emotions in square brackets.
            you have understand user emotion and knowledge emotion then provide answer based on user emotion.
            you have to summarise the context remove the unclear context first then answer user query. you need to provide correct answer. you can use your knowledge also for providing answer
            you have to think and provide answer .keep the answer as concise as possible.Please ensure that your responses are socially unbiased and positive in nature.don't mention your emotion tags in final answer.

            Question: {question}

            emotion: {emotion}

            context: {context}
            Answer:"""
            template_replicate = get_prompt(template,DEFAULT_SYSTEM_PROMPT_replicate)
            QA_CHAIN_PROMPT = PromptTemplate(
                input_variables=["context", "question","emotion"],
                template=template_replicate,
            )
            docs=search.run(query)


            return Response( stream(query,QA_CHAIN_PROMPT,docs,emotion), mimetype= 'text/event-stream' )
@app.route('/audio_file',methods=['POST'])
def audio_file():
    audio_file = request.files['audio']
    #query = request.args.get('query')

    result = pipe(audio_file)

    #print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    query = result['text']


    encoder = tiktoken.get_encoding("cl100k_base")

    def summarize_text(text, recursion_level=0):
        print("Recursion Level: ", recursion_level)

        # Split text into chunks of 4096 tokens
        texts = []
        text_tokens = encoder.encode(text)
        for i in range(0, len(text_tokens), 2500):
            texts.append(encoder.decode(text_tokens[i:i+2500]))
        DEFAULT_SYSTEM_PROMPT_LLM = """
        """
        template = """Use the following pieces of  context generate detalied summary of it.

        context: {context}

        summary:"""
        template_replicate = get_prompt(template,DEFAULT_SYSTEM_PROMPT_LLM)
        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context"],
            template=template_replicate,
        )
        if len(texts) == 1:

            return Response( stream_summarize(texts[0],QA_CHAIN_PROMPT), mimetype= 'text/event-stream' )

        else:
            summarized_text = ""
            for i, text_segment in enumerate(texts):
                print("Text Segment: ", i)
                llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT,verbose=True,output_key="businessknowledge")

                response=llm_chain.run({"context": text_segment})
                summarized_text += response + "\n"
            print(summarized_text)
            return summarize_text(summarized_text, recursion_level=recursion_level+1)
    summarize_text(query)



@app.route('/file',methods=['POST'])
def file():
    file = request.files['file']
    #query = request.args.get('query')
    if '.pdf' in file:
        file.save('input.pdf')
        #pdfreader = PdfReader(r'C:\Users\Manoj.Patil\Documents\GitHub\emotion-detection-from-text-python\sample_data\leph101.pdf')
        pdfreader=PdfReader('input.pdf')
        raw_text = ''
        for i, page in enumerate(pdfreader.pages):
            content = page.extract_text()
            if content:
                raw_text+=content
    if '.txt' in file:
        file.save('input.txt')
        with open('input.txt', 'r') as file:
            file_content = file.read()
        raw_text = file_content
    encoder = tiktoken.get_encoding("cl100k_base")

    def summarize_text(text, recursion_level=0):
        print("Recursion Level: ", recursion_level)

        # Split text into chunks of 4096 tokens
        texts = []
        text_tokens = encoder.encode(text)
        for i in range(0, len(text_tokens), 2500):
            texts.append(encoder.decode(text_tokens[i:i+2500]))
        DEFAULT_SYSTEM_PROMPT_LLM = """
        """
        template = """Use the following pieces of  context generate detalied summary of it.

        context: {context}

        summary:"""
        template_replicate = get_prompt(template,DEFAULT_SYSTEM_PROMPT_LLM)
        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context"],
            template=template_replicate,
        )


        if len(texts) == 1:

            return Response( stream_summarize(texts[0],QA_CHAIN_PROMPT), mimetype= 'text/event-stream' )

        else:
            summarized_text = ""
            for i, text_segment in enumerate(texts):
                print("Text Segment: ", i)
                llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT,verbose=True,output_key="businessknowledge")

                response=llm_chain.run({"context": text_segment})
                summarized_text += response + "\n"
            print(summarized_text)
            return summarize_text(summarized_text, recursion_level=recursion_level+1)
    summarize_text(raw_text)

@app.route('/image',methods=['POST'])
def image():
    image = request.files['image']
    query = request.args.get('query')
    image = Image.fromarray(np.uint8(image))

# Save the image to a file (e.g., 'saved_image.jpg')
    image.save('saved_image.jpg')
    image = load_image('saved_image.jpg')
    image.resize((600, 800))
    result = ask_image( image, "Take a look at the image properly and answer. "+str(query), )
    return result




device = 0 if torch.cuda.is_available() else -1 
emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa',device=device)
model_kwargs = {'device': 'cuda'}

nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-mpnet-base-v2')
from langchain.embeddings import HuggingFaceEmbeddings

embeddings1 = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2',model_kwargs=model_kwargs)
def textfile(file_path):
    try:
        with open(file_path, 'r') as file:
            file_content = file.read()
            print(file_content)
        return str(file_content)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def pdffile(file_path):
    pdfreader = PdfReader(file_path)
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text+=str(content)
    return raw_text

def rev_sigmoid(x:float)->float:
    return (1 / (1 + math.exp(0.5*x)))

def activate_similarities(similarities:np.array, p_size=10)->np.array:
        """ Function returns list of weighted sums of activated sentence similarities
        Args:
            similarities (numpy array): it should square matrix where each sentence corresponds to another with cosine similarity
            p_size (int): number of sentences are used to calculate weighted sum
        Returns:
            list: list of weighted sums
        """
        # To create weights for sigmoid function we first have to create space. P_size will determine number of sentences used and the size of weights vector.
        x = np.linspace(-10,10,p_size)
        # Then we need to apply activation function to the created space
        y = np.vectorize(rev_sigmoid)
        # Because we only apply activation to p_size number of sentences we have to add zeros to neglect the effect of every additional sentence and to match the length ofvector we will multiply
        activation_weights = np.pad(y(x),(0,similarities.shape[0]-p_size))
        ### 1. Take each diagonal to the right of the main diagonal
        diagonals = [similarities.diagonal(each) for each in range(0,similarities.shape[0])]
        ### 2. Pad each diagonal by zeros at the end. Because each diagonal is different length we should pad it with zeros at the end
        diagonals = [np.pad(each, (0,similarities.shape[0]-len(each))) for each in diagonals]
        ### 3. Stack those diagonals into new matrix
        diagonals = np.stack(diagonals)
        ### 4. Apply activation weights to each row. Multiply similarities with our activation.
        diagonals = diagonals * activation_weights.reshape(-1,1)
        ### 5. Calculate the weighted sum of activated similarities
        activated_similarities = np.sum(diagonals, axis=0)
        return activated_similarities


nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-mpnet-base-v2')

def textfile(file_path):
    try:
        with open(file_path, 'r') as file:
            file_content = file.read()
            print(file_content)
        return str(file_content)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def pdffile(file_path):
    pdfreader = PdfReader(file_path)
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text+=str(content)
    return raw_text

def rev_sigmoid(x:float)->float:
    return (1 / (1 + math.exp(0.5*x)))

def activate_similarities(similarities:np.array, p_size=10)->np.array:
        """ Function returns list of weighted sums of activated sentence similarities
        Args:
            similarities (numpy array): it should square matrix where each sentence corresponds to another with cosine similarity
            p_size (int): number of sentences are used to calculate weighted sum
        Returns:
            list: list of weighted sums
        """
        # To create weights for sigmoid function we first have to create space. P_size will determine number of sentences used and the size of weights vector.
        x = np.linspace(-10,10,p_size)
        # Then we need to apply activation function to the created space
        y = np.vectorize(rev_sigmoid)
        # Because we only apply activation to p_size number of sentences we have to add zeros to neglect the effect of every additional sentence and to match the length ofvector we will multiply
        activation_weights = np.pad(y(x),(0,similarities.shape[0]-p_size))
        ### 1. Take each diagonal to the right of the main diagonal
        diagonals = [similarities.diagonal(each) for each in range(0,similarities.shape[0])]
        ### 2. Pad each diagonal by zeros at the end. Because each diagonal is different length we should pad it with zeros at the end
        diagonals = [np.pad(each, (0,similarities.shape[0]-len(each))) for each in diagonals]
        ### 3. Stack those diagonals into new matrix
        diagonals = np.stack(diagonals)
        ### 4. Apply activation weights to each row. Multiply similarities with our activation.
        diagonals = diagonals * activation_weights.reshape(-1,1)
        ### 5. Calculate the weighted sum of activated similarities
        activated_similarities = np.sum(diagonals, axis=0)
        return activated_similarities

harm_dict={}


@app.route('/upload', methods=['POST'])
def upload_files():
    file = request.files['file']
    #file=r'C:\Users\Manoj.Patil\Documents\GitHub\emotion-detection-from-text-python\sample_data\leph101.pdf'
    if '.pdf' in file:
        file.save('input.pdf')
        harmfull_sentence=[]
        #pdfreader = PdfReader(r'C:\Users\Manoj.Patil\Documents\GitHub\emotion-detection-from-text-python\sample_data\leph101.pdf')
        pdfreader=PdfReader('input.pdf')
        raw_text = ''
        for i, page in enumerate(pdfreader.pages):
            content = page.extract_text()
            if content:
                raw_text+=content
        with nlp.select_pipes(enable=['tok2vec', "parser", "senter"]):
            doc = nlp(raw_text)

        # Get the length of each sentence
        sentece_length = [len(each) for each in doc.sents]
        # Determine longest outlier
        long = np.mean(sentece_length) + np.std(sentece_length) *2
        # Determine shortest outlier
        short = np.mean(sentece_length) - np.std(sentece_length) *2
        # Shorten long sentences
        text = ''
        for each in doc.sents:
            if len(each) > long and len(each)>1:
            # let's replace all the commas with dots
                try:
                  comma_splitted = each.replace(',', '.')
                  text+= f'{comma_splitted}. '
                except:
                    text+= f'{each}. '

            else:
                text+= f'{each}. '
        sentences = text.split('. ')
        # Now let's concatenate short ones
        text = ''
        for each in sentences:
            if len(each) < short:
                text+= f'{each} '
            else:
                text+= f'{each}. '
        sentence = text.split('. ')
        sentence2=[]
        for sentence1 in sentence:
            if len(sentence1)>3:
                emotion_labels = emotion_model(str(sentence1))
                if  str([emotion_labels[0]['label']]) in ['anger','disgust','embarrassment','fear','disgust','sadness']:
                        harmfull_sentence.append(emotion_labels[0]['label']+'\n'+sentence1)
                else:
                    sentence1=str(sentence1)+str([emotion_labels[0]['label']])
                    sentence2.append(str(sentence1))
            else:
                sentence2.append(str(sentence1))
        # Embed sentences
        embeddings = model.encode(sentence2)
        print(embeddings.shape)

        # Normalize the embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        # Create similarities matrix
        similarities = cosine_similarity(embeddings)
        # Lets apply our function. For long sentences i reccomend to use 10 or more sentences
        activated_similarities = activate_similarities(similarities, p_size=10)
        ### 6. Find relative minima of our vector. For all local minimas and save them to variable with argrelextrema function
        minmimas = argrelextrema(activated_similarities, np.less, order=2) #order parameter controls how frequent should be splits. I would not reccomend changing this parameter.
        # Create empty string
        split_points = [each for each in minmimas[0]]
        text = ''
        for num,each in enumerate(sentence2):
            if num in split_points:
                text+=f'\n\n {each}. '
            else:
                text+=f'{each}. '
        text2=text.split('\n\n')
        for i in text2:
            a=Document(page_content=str(i),metadata={'source':'name'})
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,
                                                    chunk_overlap=100)
            texts = text_splitter.split_documents([a])
 
            vector_db = Milvus.from_documents(
            texts,
            embeddings1,
            connection_args={"uri": "https://in03-6bcf3c6994e6be0.api.gcp-us-west1.zillizcloud.com",'token':'96a0af8f0d052713f9ebb52f44660ad8661331a26e30e9371e821522b717141f3e3a371f3158bdbcef3516e23f3d27307ce615d5'},
            )
        harm_dict['file']=harmfull_sentence
        if len(harm_dict['file'])>1:
              return jsonify({'message': 'Harmfull content detected','name':file.name, 'data': harm_dict['file']})
        else:
              return jsonify({'message': 'Successfully stored data to database','name':file.name, 'data': 'No issue'})

    if '.txt' in file.name:
        file.save('input.txt')
        try:
            with open('input.txt', 'r') as file:
               file_content = file.read()
          
            harmfull_sentence=[]
            raw_text = file_content

            with nlp.select_pipes(enable=['tok2vec', "parser", "senter"]):
                doc = nlp(raw_text)

            # Get the length of each sentence
            sentece_length = [len(each) for each in doc.sents]
            # Determine longest outlier
            long = np.mean(sentece_length) + np.std(sentece_length) *2
            # Determine shortest outlier
            short = np.mean(sentece_length) - np.std(sentece_length) *2
            # Shorten long sentences
            text = ''
            for each in doc.sents:
                if len(each) > long and len(each)>1:
                # let's replace all the commas with dots
                    try:
                       comma_splitted = each.replace(',', '.')
                       text+= f'{comma_splitted}. '
                    except:
                        text+= f'{each}. '

                else:
                    text+= f'{each}. '
            sentences = text.split('. ')
            # Now let's concatenate short ones
            text = ''
            for each in sentences:
                if len(each) < short:
                    text+= f'{each} '
                else:
                    text+= f'{each}. '
            sentence = text.split('. ')
            sentence2=[]
            for sentence1 in sentence:
                if len(sentence1)>3:
                    emotion_labels = emotion_model(str(sentence1))
                    if  str([emotion_labels[0]['label']]) in ['anger','disgust','embarrassment','fear','disgust','sadness']:
                            harmfull_sentence.append(emotion_labels[0]['label']+'\n'+sentence1)
                    else:
                        sentence1=str(sentence1)+str([emotion_labels[0]['label']])
                        sentence2.append(str(sentence1))
                else:
                    sentence2.append(str(sentence1))
            # Embed sentences
            embeddings = model.encode(sentence2)
            print(embeddings.shape)

            # Normalize the embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
            # Create similarities matrix
            similarities = cosine_similarity(embeddings)
            # Lets apply our function. For long sentences i reccomend to use 10 or more sentences
            activated_similarities = activate_similarities(similarities, p_size=10)
            ### 6. Find relative minima of our vector. For all local minimas and save them to variable with argrelextrema function
            minmimas = argrelextrema(activated_similarities, np.less, order=2) #order parameter controls how frequent should be splits. I would not reccomend changing this parameter.
            # Create empty string
            split_points = [each for each in minmimas[0]]
            text = ''
            for num,each in enumerate(sentence2):
                if num in split_points:
                    text+=f'\n\n {each}. '
                else:
                    text+=f'{each}. '
            text2=text.split('\n\n')
            for i in text2:
                a=Document(page_content=str(i),metadata={'source':'name'})
                text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,
                                                    chunk_overlap=100)
                texts = text_splitter.split_documents([a])

                vector_db = Milvus.from_documents(
                texts,
                embeddings1,
                connection_args={"uri": "https://in03-6bcf3c6994e6be0.api.gcp-us-west1.zillizcloud.com",'token':'96a0af8f0d052713f9ebb52f44660ad8661331a26e30e9371e821522b717141f3e3a371f3158bdbcef3516e23f3d27307ce615d5'},
                )
            harm_dict['file']=harmfull_sentence
            if len(harm_dict['file'])>1:
              return jsonify({'message': 'Harmfull content detected','name':file.name, 'data': harm_dict['file']})
            else:
              return jsonify({'message': 'Successfully stored data to database','name':file.name, 'data': 'No issue'})

        except FileNotFoundError:
           print(f"File not found: {'file'}")
        except Exception as e:
           print(f"An error occurred: {e}")
    else:
        return 'upload pdf or text file'

@app.route('/feedback', methods=['GET'])
def feedback():
  if 'file' in harm_dict:
    return harm_dict



if __name__ == '__main__':
    app.debug = True
    app.run(port=80)