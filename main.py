import csv
import hashlib
import json
import logging
import time
import chainlit as cl

from functools import lru_cache
from pathlib import Path
from typing import path
from os import path, getenv
from dotenv import load_dotenv
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import PromptTemplate, SimpleDirectoryReader, VectorStoreIndex,Settings

_ = load_dotenv()
logging.basicConfig(level=getenv("LOG_LEVEL","INFO"))
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR /"AquivosCSV"
CSV_PATH =DATA_DIR/"sales.csv"
STORAGE_DIR = BASE_DIR / "storage"
INDEX_METADATA_PATH = STORAGE_DIR / "index_metadata.json"

QA_TEMPLATE = PromptTemplate("""

Você é um assistente especializado em consultar uma base de vendas.

Regras:
- Responda sempre em português do Brasil.
- Use apenas informações do contexto.
- Se a pergunta for sobre preço/valor de produto, responda com o preço unitário quando existir.
- Se houver mais de uma venda do mesmo produto, explique que o produto aparece em mais de uma venda.
- Se a pergunta envolver quantidade ou total vendido, use os campos quantidade e valor_total.
- Se a informação não estiver no contexto, diga que não encontrou a informação na base.

Contexto:
{context_str}

Pergunta:
{query_str}

Resposta final:

""")

SYSTEM_PROMPT = """"
Você é um assistente técnico diretp, preciso e útil.
Responda sempre em português do Brasil.
Quando não souber a resposta, diga que não sabe.
"""



def get_required_env(name: str) -> str:
    value = getenv(name)
    if not value:
        raise ValueError(f"Variável de ambiente '{name}' não encontrada.")
    return value

def create_llm()-> LlamaCPP:
    return LlamaCPP(
        model_path=get_required_env("caminho"),
        temperature=0.1,
        max_new_tokens=512,
        # context_window=3900, # Descomente e ajuste conforme a capacidade da sua RAM
        model_kwargs={"n_gpu_layers": -1}, # Descomente se tiver placa de vídeo para acelerar
    )

def build_prompt(history: list[dict[str:object]],user_message:str)->PromptTemplate:
    recent_messages = history[-8:]  # Pega as últimas 3 mensagens
    conversation ="\n".join(
        f"{item['role']}: {item['content']}" for item in recent_messages
    )
    PROMPT=(
        "{system_prompt}\n"
        "Histórico recente da conversa: {conversation}\n"
        "User: {user_message}\n"
        "Assistant:".strip()
    )
    return PromptTemplate(
        prompt=PROMPT,
        system_prompt=SYSTEM_PROMPT,
        conversation=conversation,
        user_message=user_message
    )

@cl.on_chat_start
async def on_chat_start()-> None:
    cl.user_session.set("llm", create_llm())
    cl.user_session.set("embed_model", HuggingFaceEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))
    cl.user_session.set("history", [])
    await cl.Message(content="Chat inicado. pode enviar sau pergunta.").send()

@cl.on_message
async def main(message: cl.Message)-> None:
    llm = cl.user_session.get("llm") or create_llm()
    embed_model = cl.user_session.get("embed_model")
    history = cl.user_session.get("history") or []
    prompt = build_prompt(history,message.content)

    msg = cl.Message(content="")
    await msg.send()

    answer =""
    async for token in llm.stream(prompt):
        answer += token
        await msg.stream_token(token)
    history.append({"role":"Usuãrio","content":message.content})
    history.append({"role":"Assistant","content":answer})
    
    cl.user_session.set("history",history[-12:])

    msg.content = answer 
    await msg.update()
    # response = llm(prompt)
    # await cl.Message(content=response).send()

