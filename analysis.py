from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import re

# Extract video ID
def extract_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    return match.group(1) if match else None

# Document Ingestion
def get_video_transcript(video_id = "Gfr50f6ZBvo", language="en"):
  try:
      # If you don’t care which language, this returns the “best” one
      transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])

      # Flatten it to plain text
      transcript = " ".join(chunk["text"] for chunk in transcript_list)
    #   print(transcript)
      return transcript

  except TranscriptsDisabled:
      print("No captions available for this video.")

# Text Splitting
def split_text(transcript):
   splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
   chunks = splitter.create_documents([transcript])
   return chunks

# Embedding Generation and Storing in Vector Store
def generate_embedding(chunks):
   embeddings  = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
   vector_store = FAISS.from_documents(chunks, embeddings)
   return vector_store

# Retrieval
def get_retreiver(vector_store, search_type="similarity", k=4):
   retriever = vector_store.as_retriever(search_type=search_type, search_kwargs={"k": k})
   return retriever

load_dotenv()  # loads .env file

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

def get_answer(video_id = "KKNCiRWd_j0", language="en", search_type="similarity", k=4, query='Can you summarize the video'):
   transcript = get_video_transcript(video_id = video_id, language=language)
   chunks = split_text(transcript)
   vector_store = generate_embedding(chunks)
   retriever = get_retreiver(vector_store, search_type=search_type, k=k)

   # Augmentation
   llm = ChatGroq(model="llama3-8b-8192")

   prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
    )
   
   # Building a Chain
   parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
    })
   
   parser = StrOutputParser()

   main_chain = parallel_chain | prompt | llm | parser

   # Generation
   ans = main_chain.invoke(query)

   return ans

if __name__ == "__main__" :
   res = get_answer(video_id = "KKNCiRWd_j0", language="en", search_type="similarity", k=4, query='Can you summarize the video')
   print(res)
   
   
















