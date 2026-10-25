from os import path, getenv
import chainlit as cl
from dotenv import load_dotenv
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex,Settings

_ = load_dotenv()
SYSTEM_PROMPT = """"
Você é um assistente técnico diretp, preciso e útil.
Responda sempre em português do Brasil.
Quando não souber a resposta, diga que não sabe.
"""
caminho_modelo_llm = getenv("caminho")
# caminho_modelo_llm = r"C:\Users\sydhi\.cache\lm-studio\models\lmstudio-community\Qwen3-14B-GGUF\.gguf"

# 2. Configurar o LLM usando LlamaCPP
llm = LlamaCPP(
    model_path=caminho_modelo_llm,
    temperature=0.1,
    max_new_tokens=256,
    # context_window=3900, # Descomente e ajuste conforme a capacidade da sua RAM
    # model_kwargs={"n_gpu_layers": -1}, # Descomente se tiver placa de vídeo para acelerar
)
# Settings.embed_model = HuggingFaceEmbedding(
#     model_name="BAAI/bge-small-en-v1.5"
# )
caminho_modelo_embed = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" 
# Se não tiver baixado, pode usar a string: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# 3. Aplicar as configurações ao LlamaIndex
Settings.llm = llm
Settings.embed_model = embed_model # <-- Isso remove a dependência da OpenAI
diretorio = path.dirname(path.abspath(__file__))
caminho_csv = path.join(diretorio, 'AquivosCSV')

documents = SimpleDirectoryReader(caminho_csv).load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

response = query_engine.query("Qual é o produto que custa 4500?  Responda detalhadamente em português.")
print(response)