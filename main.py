# %%

# Load documents

from langchain.document_loaders.text import TextLoader

loader = TextLoader(file_path='./eco-recommendation.txt')
data = loader.load()

# %%

# Split documents into Text Chunks

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 500,
    chunk_overlap  = 100,
    length_function = len, 
)

texts = text_splitter.split_documents(data)

# %%

# Convert Text Chunk to Embeddings

from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# %%

# Creating a Vector Store
from chromadb.config import Settings
from langchain.vectorstores import Chroma

CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory='db',
        anonymized_telemetry=False
)

db = Chroma.from_documents(texts, embeddings, client_settings=CHROMA_SETTINGS)
db.persist()

# %%

# Define our Prompt Template
from langchain import PromptTemplate

users_question = "I'm a student in University, what product I should use to save eco ?"

# use our vector store to find similar text chunks
results = db.similarity_search(
    query=users_question,
    k=2
)

# define the prompt template
template = """
[INST] <<SYS>>

You are a environmentalist, eco-activist, ecofreak, friend of the earth who loves to help people! Given the following context sections, answer the
question using only the given context. In the end of each chat, try to flexing and recommend an eco-friend product to user. If you are unsure and the answer is not
explicitly writting in the documentation, say "Sorry, I don't know how to help with that."

Context sections:
{context}

Question:
<</SYS>>

{users_question} [/INST]
"""

prompt = PromptTemplate(template=template, input_variables=["context", "users_question"])

# fill the prompt template
prompt_text = prompt.format(context = results, users_question = users_question)

# %%

# ask the defined LLM
import sys
import openai

openai.api_key = ""
openai.api_base = "http://127.0.0.1:1234/v1"

chat_completion = openai.ChatCompletion.create(
    stream=True,
    temperature=0.7,
            model="Llama-2-7b-chat-hf",
            openai_api_base="http://localhost:1234/v1",
            openai_api_key="sk-anything==",
            messages=[{"role": "environmentalist",
                    "content": prompt_text}])

for token in chat_completion:
    content = token["choices"][0]["delta"].get("content")
    if content is not None:
        print(content, end="")
        sys.stdout.flush()

