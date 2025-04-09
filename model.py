import os
import torch
from dotenv import load_dotenv
from transformers import pipeline

# LangChain imports
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# LLM Providers
from langchain_community.llms import CTransformers, HuggingFacePipeline, HuggingFaceHub, Replicate, Anthropic, Cohere
from langchain.llms import OpenAI
from langchain_google_vertexai import VertexAI
import chainlit as cl

# ------------------------------
# Load environment variables
# ------------------------------
load_dotenv()
hf_token = os.getenv("HF_TOKEN")  # Hugging Face access token

# ------------------------------
# Device Configuration
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------
# Set paths
# ------------------------------
DB_FAISS_PATH = "vectorstore/db_faiss_ML"

# ------------------------------
# Custom Prompt Template
# ------------------------------
custom_prompt_template = """SashaktAI is committed to providing exceptional service in a respectful and truthful manner. My responses will be clear, concise, and free of bias.

**Context:** {context}
**Question:** {question}

If there is nothing available in Context, politely say that you don't know the answer and do not make one up.
Helpful answer:
"""

def set_custom_prompt():
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )

# ------------------------------
# Load Hugging Face Embedding Model
# ------------------------------
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

# ------------------------------
# Load FAISS Vector Store
# ------------------------------
def load_vector_db(embeddings):
    return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

# ------------------------------
# Load LLMs
# ------------------------------

def load_llm(provider="ctransformers"):
    hf_token = os.getenv("HF_TOKEN")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if provider == "ctransformers":
        config = {
            "max_new_tokens": 512,
            "context_length": 8000
        }
        return CTransformers(
            model="TheBloke/Llama-2-7B-Chat-GGUF",
            model_type="llama",
            temperature=0,
            device="cuda",
            config=config,
        )

    elif provider == "huggingface_pipeline":
        hf_pipeline = pipeline(
            "text-generation",
            model="tiiuae/falcon-7b-instruct",
            device=0
        )
        return HuggingFacePipeline(pipeline=hf_pipeline)

    elif provider == "openai":
        return OpenAI(
            model_name="gpt-3.5-turbo",
            openai_api_key=openai_api_key,
            temperature=0,
            max_tokens=512,
        )

    elif provider == "huggingface_hub":
        return HuggingFaceHub(
            repo_id="bigscience/bloom",
            huggingfacehub_api_token=hf_token,
            model_kwargs={"temperature": 0.5, "max_length": 512}
        )

    elif provider == "replicate":
        return Replicate(
            model="a16z-infra/llama13b-v2-chat",
            input={"temperature": 0.75, "max_length": 512}
        )

    elif provider == "anthropic":
        return Anthropic(
            model="claude-2",
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens_to_sample=512
        )

    elif provider == "cohere":
        return Cohere(
            model="command-xlarge-nightly",
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            temperature=0.5,
            max_tokens=512
        )

    elif provider == "vertexai":
        return VertexAI(
            model_name="text-bison",
            temperature=0.3,
            max_output_tokens=512
        )

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


# ------------------------------
# Build the RAG QA Chain
# ------------------------------
def build_qa_chain():
    embeddings = load_embeddings()
    db = load_vector_db(embeddings)
    llm = load_llm(provider="ctransformers")
    prompt = set_custom_prompt()

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_rerank",
        retriever=db.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

# ------------------------------
# Chainlit Handlers
# ------------------------------
@cl.on_chat_start
async def start():
    chain = build_qa_chain()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = """**Welcome to SashkatAI!**\nI'm your AI-powered research assistant for scientific documents. Ask me anything!"""
    await msg.update()
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    await cl.Message(content=res["result"]).send()