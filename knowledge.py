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


def extract_document_info(doc_name, text):
    """
    Extrai informações estruturais do documento como tipo, número e data.
    
    Args:
        doc_name (str): Nome do documento
        text (str): Texto do documento
    
    Returns:
        dict: Metadados extraídos
    """
    info = {
        "source": doc_name,
        "doc_type": "unknown"
    }
    
    # Identifica o tipo de documento
    if "lei" in doc_name.lower():
        info["doc_type"] = "lei"
    elif "decreto" in doc_name.lower():
        info["doc_type"] = "decreto"
    elif "resolução" in doc_name.lower() or "resolucao" in doc_name.lower():
        info["doc_type"] = "resolucao"
    elif "código" in doc_name.lower() or "codigo" in doc_name.lower():
        info["doc_type"] = "codigo"
    
    # Extrai número do documento
    number_match = re.search(r'(?:n[º°.]?\s*)([\d\.]+)(?:/(\d{4}))?', doc_name)
    if number_match:
        info["doc_number"] = number_match.group(1)
        if number_match.group(2):  # Ano
            info["doc_year"] = number_match.group(2)
    
    return info


def split_legal_text(text, doc_info):
    """
    Divide o texto legal em chunks baseados na estrutura de artigos/seções.
    
    Args:
        text (str): Texto limpo do documento
        doc_info (dict): Informações estruturais do documento
    
    Returns:
        list: Lista de chunks com metadados
    """
    # Verifica se doc_info é um dicionário
    if not isinstance(doc_info, dict):
        doc_info = {"source": "unknown"}
    
    chunks = []
    
    # Padrões para documentos jurídicos brasileiros
    article_pattern = r'Art\.?\s*(\d+[º°]?[A-Z]?)[.\s-]+(.*?)(?=Art\.?\s*\d+[º°]?[A-Z]?|$)'
    
    # Tenta dividir por artigos
    articles = re.findall(article_pattern, text, re.DOTALL)
    
    if articles:
        for number, content in articles:
            # Limpa o conteúdo
            clean_content = clean_text(content)
            
            # Cria metadados específicos para este chunk
            metadata = doc_info.copy()
            metadata["article_number"] = number.strip()
            metadata["content_type"] = "article"
            
            # Cria o chunk com referência explícita ao artigo
            chunk_text = f"Artigo {number.strip()}: {clean_content}"
            
            # Se o artigo for muito grande, subdivide
            if len(chunk_text) > CHUNK_SIZE:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP
                )
                sub_chunks = text_splitter.split_text(chunk_text)
                
                for i, sub_chunk in enumerate(sub_chunks):
                    sub_metadata = metadata.copy()
                    sub_metadata["part"] = i + 1
                    sub_metadata["total_parts"] = len(sub_chunks)
                    
                    # Adiciona como dicionário com text e metadata
                    chunks.append({
                        "text": sub_chunk,
                        "metadata": sub_metadata
                    })
            else:
                # Artigo não é grande, manter como um único chunk
                chunks.append({
                    "text": chunk_text,
                    "metadata": metadata
                })
    else:
        # Se não encontrou estrutura de artigos, usa chunking padrão
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        simple_chunks = text_splitter.split_text(text)
        
        for i, chunk in enumerate(simple_chunks):
            metadata = doc_info.copy()
            metadata["chunk_index"] = i
            metadata["total_chunks"] = len(simple_chunks)
            
            # Adiciona como dicionário com text e metadata
            chunks.append({
                "text": chunk,
                "metadata": metadata
            })
    
    return chunks