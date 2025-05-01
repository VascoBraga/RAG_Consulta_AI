import os
from dotenv import load_dotenv

def load_environment():
    """Carrega variáveis de ambiente e verifica a API key."""
    load_dotenv()
    
    # Verifica se a API key está configurada
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("A API key da OpenAI não está configurada. Adicione-a ao arquivo .env")

# Configurações globais
EMBED_MODEL = "text-embedding-ada-002"  # Modelo de embeddings
LLM_MODEL = "gpt-3.5-turbo"             # Modelo LLM
CHUNK_SIZE = 1000                       # Tamanho dos chunks de texto
CHUNK_OVERLAP = 200                     # Sobreposição entre chunks
PDF_PATH = "CDC_2024.pdf"               # Caminho do arquivo PDF
DB_PATH = "chroma_db"                   # Diretório para o banco de dados Chroma