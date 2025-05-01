import os
import re
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from init import EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, PDF_PATH, DB_PATH

def extract_text_from_pdf(pdf_path):
    """Extrai texto do arquivo PDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def clean_text(text):
    """Limpa o texto removendo espaços extras, caracteres especiais, etc."""
    # Substitui múltiplos espaços em branco por um único espaço
    text = re.sub(r'\s+', ' ', text)
    # Remove caracteres de controle e outros não imprimíveis
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    return text.strip()

def split_text(text):
    """Divide o texto em chunks para processamento."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_db(chunks):
    """Gera embeddings e armazena no Chroma DB."""
    # Inicializa o modelo de embeddings
    embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL
        # Removido o parâmetro dimensions que não é mais suportado
    )
    
    # Cria o banco de dados Chroma
    db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    
    # Persiste o banco de dados
    db.persist()
    return db

def load_vector_db():
    """Carrega o banco de dados vetorial existente."""
    embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL
        # Removido o parâmetro dimensions que não é mais suportado
    )
    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
    return db

def process_pdf_and_create_db():
    """Processa o PDF e cria o banco de dados vetorial."""
    print("Extraindo texto do PDF...")
    raw_text = extract_text_from_pdf(PDF_PATH)
    
    print("Limpando o texto...")
    cleaned_text = clean_text(raw_text)
    
    print("Dividindo o texto em chunks...")
    chunks = split_text(cleaned_text)
    print(f"Texto dividido em {len(chunks)} chunks.")
    
    print("Gerando embeddings e armazenando no Chroma DB...")
    db = create_vector_db(chunks)
    print(f"Banco de dados vetorial criado com sucesso em '{DB_PATH}'.")
    
    return db

def get_or_create_db():
    """Verifica se o banco de dados existe e carrega-o ou cria-o se necessário."""
    if os.path.exists(DB_PATH) and len(os.listdir(DB_PATH)) > 0:
        print(f"Carregando banco de dados vetorial existente de '{DB_PATH}'...")
        return load_vector_db()
    else:
        print("Banco de dados vetorial não encontrado. Criando um novo...")
        return process_pdf_and_create_db()