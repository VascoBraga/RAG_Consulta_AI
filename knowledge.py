import os
import re
import fitz  # PyMuPDF
from io import BytesIO 
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
from init import EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, DB_PATH, DOCUMENTO_PATHS

def extract_text_from_pdf(pdf_path):
    """Extrai texto do arquivo PDF com tratamento de erro robusto."""
    # Verificação do arquivo
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {pdf_path}")
    
    print(f"Abrindo arquivo: {pdf_path}")
    
    try:
        # Abrir o PDF como um arquivo binário primeiro
        with open(pdf_path, 'rb') as file_handle:
            # Passar o conteúdo binário para o fitz
            pdf_data = file_handle.read()
            memory_stream = BytesIO(pdf_data)
            doc = fitz.open(stream=memory_stream, filetype="pdf")
            
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
    except Exception as e:
        print(f"Erro ao processar o PDF {pdf_path}: {str(e)}")
        # Tente um método alternativo
        try:
            import PyPDF2
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num in range(len(reader.pages)):
                    text += reader.pages[page_num].extract_text()
            return text
        except Exception as e2:
            print(f"Método alternativo também falhou: {str(e2)}")
            raise

def clean_text(text):
    """Limpa o texto removendo espaços extras, caracteres especiais, etc."""
    # Substitui múltiplos espaços em branco por um único espaço
    text = re.sub(r'\s+', ' ', text)
    # Remove caracteres de controle e outros não imprimíveis
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    return text.strip()


def split_text(text, doc_info=None):
    """Divide o texto em chunks para processamento."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Isso retorna uma lista de strings
    raw_chunks = text_splitter.split_text(text)
    print(f"Texto dividido em {len(raw_chunks)} raw_chunks (strings)")
    
    # Convertemos para o formato esperado com metadados
    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        # Verificação de segurança - garantir que o texto é uma string
        if not isinstance(chunk_text, str):
            print(f"AVISO: chunk_text não é uma string, é {type(chunk_text)}")
            chunk_text = str(chunk_text)
        
        # Criamos um dicionário para cada chunk
        chunk = {
            "text": chunk_text,
            "metadata": {
                "chunk_index": i,
                "total_chunks": len(raw_chunks)
            }
        }
        
        # Adicionamos metadados do documento, se fornecidos
        if doc_info and isinstance(doc_info, dict):
            chunk["metadata"].update(doc_info)
        
        chunks.append(chunk)
    
    # Verificação final
    print(f"Convertido para {len(chunks)} chunks (dicionários)")
    

    print(f"DEBUGGING split_text: Retornando {len(chunks)} chunks")
    for i in range(min(3, len(chunks))):
        print(f"DEBUGGING split_text: Chunk {i} é do tipo {type(chunks[i])}")

    return chunks


def create_vector_db(chunks):
    """Gera embeddings e armazena no Chroma DB."""
    # Inicializa o modelo de embeddings
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    
    # Processa chunks para extrair textos e metadados de forma segura
    texts = []
    metadatas = []
    
    for chunk in chunks:
        if isinstance(chunk, dict) and "text" in chunk and "metadata" in chunk:
            texts.append(chunk["text"])
            metadatas.append(filter_complex_metadata(chunk["metadata"]))
        elif isinstance(chunk, str):
            texts.append(chunk)
            metadatas.append({})
        else:
            print(f"AVISO: Tipo de chunk inesperado: {type(chunk)}")
            continue
    
    # Cria o banco de dados Chroma
    db = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=DB_PATH
    )
    
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

def get_or_create_db():
    """Verifica se o banco de dados existe e carrega-o ou cria-o se necessário."""
    if os.path.exists(DB_PATH) and len(os.listdir(DB_PATH)) > 0:
        print(f"Carregando banco de dados vetorial existente de '{DB_PATH}'...")
        return load_vector_db()
    else:
        print("Banco de dados vetorial não encontrado. Criando um novo...")
        return process_all_documents()

def extract_article_metadata(text):
    """Extrai metadados. Ex: Número do artigo, capítulo etc."""
    metadata = {}

    #Padrões para identificar elementos estruturais
    article_match = re.search(r'Art\.\s*(d+)', text)
    chapter_match = re.search(r'CAPÍTULO\s+([IVX]+[0-9]+)', text, re.IGNORECASE)
    title_match   = re.search(r'TÍTULO\s+(IVX]+|[0-9]+)', text, re.IGNORECASE)

    if article_match:
        metadata['article'] = article_match.group(1)
    if chapter_match:
        metadata['chapter'] = chapter_match.group(1)
    if title_match:
        metadata['title'] = article_match.group(1)
    
    return metadata

def split_text_by_articles(text):
    """Divide o texto em chunks baseados na estrutura de artigos do CDC."""
    # Regex para identificar padrões de artigos no CDC
    article_pattern = r'Art\.\s*(\d+)\.?\s*(.?)(?=Art\.\s\d+\.?|$)'
    
    # Encontra todos os artigos
    articles = re.findall(article_pattern, text, re.DOTALL)
    chunks = []
    
    for number, content in articles:
        # Limpa o conteúdo
        clean_content = clean_text(content)
        
        # Cria um chunk com metadados
        chunk = {
            "text": f"Artigo {number}: {clean_content}",
            "metadata": {
                "source": "CDC",
                "article_number": number,
                "content_type": "article"
            }
        }
        chunks.append(chunk)
        
        # Se o artigo for muito grande, divide em sub-chunks
        if len(clean_content) > CHUNK_SIZE:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            sub_chunks = text_splitter.split_text(clean_content)
            
            # Adiciona cada sub-chunk com metadados preservados
            for i, sub_chunk in enumerate(sub_chunks):
                sub_chunk_dict = {
                    "text": f"Artigo {number} (parte {i+1}): {sub_chunk}",
                    "metadata": {
                        "source": "CDC",
                        "article_number": number,
                        "content_type": "article_part",
                        "part_number": i+1
                    }
                }
                chunks.append(sub_chunk_dict)
    
    return chunks


def update_vector_db(document_paths, vector_db=None, document_metadata=None):
    """
    Atualiza o banco de dados existente com novos documentos.
    """
    if vector_db is None:
        vector_db = get_or_create_db()
    
    if not hasattr(vector_db, 'add_texts'):
        print("Criando um novo banco de dados vetorial...")
        embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
        vector_db = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embeddings
        )
    
    if document_metadata is None:
        document_metadata = {}
    
    total_chunks = 0
    
    for doc_name, doc_path in document_paths.items():
        if not os.path.exists(doc_path):
            print(f"AVISO: Arquivo não encontrado: {doc_path}. Pulando...")
            continue
            
        print(f"Processando documento: {doc_name} ({doc_path})")
        
        try:
            # Extrai e processa o texto
            if doc_path.lower().endswith('.pdf'):
                raw_text = extract_text_from_pdf(doc_path)
            elif doc_path.lower().endswith(('.txt', '.md')):
                with open(doc_path, 'r', encoding='utf-8') as file:
                    raw_text = file.read()
            else:
                print(f"Formato de arquivo não suportado para {doc_path}. Pulando...")
                continue
            
            # Limpa e processa o texto
            cleaned_text = clean_text(raw_text)
            
            # Extrai informação estrutural do documento
            doc_info = extract_document_info(doc_name, cleaned_text)
            
            # Adiciona metadados personalizados, se disponíveis
            if doc_name in document_metadata:
                doc_info.update(document_metadata[doc_name])
            
            # Divide o texto em chunks
            chunks = split_legal_text(cleaned_text, doc_info)
            
            print(f"  -> Documento dividido em {len(chunks)} chunks.")
            total_chunks += len(chunks)
            
            # ABORDAGEM SIMPLIFICADA: Processa chunks diretamente
            texts = []
            metadatas = []
            
            # Coleta todos os textos e metadados primeiro
            for chunk in chunks:
                # Usamos apenas os chunks que são dicionários com formato correto
                if isinstance(chunk, dict) and "text" in chunk and "metadata" in chunk:
                    text = chunk["text"]
                    metadata = chunk["metadata"]
                    
                    # Verifica se metadata é um dicionário
                    if isinstance(metadata, dict):
                        texts.append(text)
                        metadatas.append(filter_complex_metadata(metadata))
            
            # Verifica se temos dados para adicionar
            if texts and metadatas:
                # Adiciona todos de uma vez
                vector_db.add_texts(texts=texts, metadatas=metadatas)
                print(f"  -> {len(texts)} chunks adicionados ao banco de dados com sucesso.")
            else:
                print("  -> Nenhum chunk válido para adicionar ao banco de dados.")
                
        except Exception as e:
            print(f"Erro ao processar documento {doc_name}: {str(e)}")
            print(f"Continuando com os próximos documentos...")
    
    print(f"Banco de dados atualizado com sucesso. Total de {total_chunks} novos chunks adicionados.")
    return vector_db


def month_to_number(month_name):
    """Converte nome do mês para número."""
    months = {
        'janeiro': '01', 'fevereiro': '02', 'março': '03', 'abril': '04',
        'maio': '05', 'junho': '06', 'julho': '07', 'agosto': '08',
        'setembro': '09', 'outubro': '10', 'novembro': '11', 'dezembro': '12'
    }
    return months.get(month_name.lower(), '00')



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
    
    # Tenta extrair a data de promulgação do texto
    date_match = re.search(r'(\d{1,2})\s+de\s+([a-zç]+)\s+de\s+(\d{4})', text, re.IGNORECASE)
    if date_match:
        info["publication_date"] = f"{date_match.group(1)}/{month_to_number(date_match.group(2))}/{date_match.group(3)}"
    
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
            metadata = doc_info.copy() if isinstance(doc_info, dict) else {"source": "document"}
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
            metadata = doc_info.copy() if isinstance(doc_info, dict) else {"source": "document"}
            metadata["chunk_index"] = i
            metadata["total_chunks"] = len(simple_chunks)
            
            # Adiciona como dicionário com text e metadata
            chunks.append({
                "text": chunk,
                "metadata": metadata
            })
    
    # Verificação de segurança adicional
    validated_chunks = []
    for chunk in chunks:
        if isinstance(chunk, dict) and "text" in chunk and "metadata" in chunk:
            validated_chunks.append(chunk)
        elif isinstance(chunk, str):
            # Se por algum motivo ainda tivermos strings, convertemos para o formato correto
            validated_chunks.append({
                "text": chunk,
                "metadata": {"source": "document"}
            })
    
    print(f"DEBUGGING split_legal_text: Retornando {len(chunks)} chunks")
    for i in range(min(3, len(chunks))):
        print(f"DEBUGGING split_legal_text: Chunk {i} é do tipo {type(chunks[i])}")

    return validated_chunks


def integrate_consumer_law_documents():
    """Integra documentos de legislação do consumidor - implementação direta."""
    # Carrega o banco de dados existente ou cria um novo
    vector_db = None
    if os.path.exists(DB_PATH) and len(os.listdir(DB_PATH)) > 0:
        print(f"Carregando banco de dados vetorial existente de '{DB_PATH}'...")
        vector_db = load_vector_db()
    
    # Metadados para o CDC
    cdc_metadata = {
        "description": "Lei principal que estabelece normas de proteção e defesa do consumidor",
        "importance": "alta",
        "category": "direitos_basicos",
        "publication_date": "11/09/1990",
        "scope": "geral",
        "hierarchy": "lei_principal"
    }
    
    # Processa o documento do CDC
    cdc_path = DOCUMENTO_PATHS["Código do Consumidor (Lei nº 8.078/90)"]
    vector_db = process_and_add_document(
        "Código do Consumidor (Lei nº 8.078/90)",
        cdc_path,
        vector_db=vector_db, 
        custom_metadata=cdc_metadata
    )
    
    # Adicione outros documentos conforme necessário
    
    return vector_db


def configure_advanced_retriever(db):
    """Configura um retriever avançado que aproveita os metadados dos documentos."""
    
    # Retriever base com MMR para maior diversidade
    base_retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 15,
            "lambda_mult": 0.5
        }
    )
    
    # Função para reranquear os resultados com base em metadados
    def rerank_by_metadata(query, docs):
        # Adiciona pontuação adicional com base nos metadados
        for doc in docs:
            base_score = doc.metadata.get("score", 0)
            
            # Bônus para documentos mais importantes
            if doc.metadata.get("importance") == "alta":
                base_score += 0.2
            
            # Bônus para artigos específicos vs. partes genéricas
            if doc.metadata.get("content_type") == "article":
                base_score += 0.1
            
            # Bônus para documentos mais recentes
            if doc.metadata.get("doc_year"):
                try:
                    year = int(doc.metadata["doc_year"])
                    # Documentos dos últimos 5 anos recebem bônus
                    if year > 2018:
                        base_score += 0.1
                except:
                    pass
            
            # Atualiza o score
            doc.metadata["adjusted_score"] = base_score
        
        # Reordena com base no novo score
        sorted_docs = sorted(docs, key=lambda x: x.metadata.get("adjusted_score", 0), reverse=True)
        return sorted_docs
    
    # Função wrapper para o retriever que aplica reranking
    def advanced_retriever(query):
        docs = base_retriever.get_relevant_documents(query)
        return rerank_by_metadata(query, docs)
    
    return advanced_retriever


def process_and_add_document(doc_name, doc_path, vector_db=None, custom_metadata=None):
    """
    Processa um único documento e o adiciona ao banco de dados - abordagem simples e direta.
    """
    print(f"Processando documento: {doc_name} ({doc_path})")
    
    # Obtém ou cria o banco de dados
    if vector_db is None:
        vector_db = load_vector_db() if os.path.exists(DB_PATH) and len(os.listdir(DB_PATH)) > 0 else None
    
    if vector_db is None or not hasattr(vector_db, 'add_texts'):
        embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
        vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    
    # Extrai e limpa o texto
    raw_text = extract_text_from_pdf(doc_path)
    cleaned_text = clean_text(raw_text)
    
    # Divide o texto em partes menores
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(cleaned_text)
    print(f"Documento dividido em {len(chunks)} chunks")
    
    # Prepara metadados base
    base_metadata = {
        "source": doc_name,
        "document_path": doc_path
    }
    
    # Adiciona metadados personalizados se fornecidos
    if custom_metadata and isinstance(custom_metadata, dict):
        base_metadata.update(custom_metadata)
    
    # Filtra metadados complexos usando nossa função personalizada
    base_metadata = filter_metadata_dict(base_metadata)
    
    # Prepara listas de textos e metadados
    texts = []
    metadatas = []
    
    for i, chunk in enumerate(chunks):
        # Garantindo que chunk é uma string
        if not isinstance(chunk, str):
            chunk = str(chunk)
        
        # Cria metadados para este chunk
        metadata = base_metadata.copy()
        metadata["chunk_index"] = i
        metadata["total_chunks"] = len(chunks)
        
        # Adiciona às listas
        texts.append(chunk)
        metadatas.append(metadata)
    
    # Adiciona ao banco de dados
    if texts and metadatas:
        vector_db.add_texts(texts=texts, metadatas=metadatas)
        print(f"Adicionados {len(texts)} chunks ao banco de dados")
    
    return vector_db


def process_all_documents():
    """Processa todos os documentos e os adiciona ao banco de dados."""
    print("Iniciando processamento de todos os documentos...")
    
    # Inicializa modelo de embeddings
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    
    # Lista para armazenar todos os chunks com metadados
    all_texts = []
    all_metadatas = []
    
    # Processa cada documento
    for doc_name, doc_path in DOCUMENTO_PATHS.items():
        print(f"\nProcessando documento: {doc_name}")
        try:
            if os.path.exists(doc_path):
                # Extrai e processa o texto
                raw_text = extract_text_from_pdf(doc_path)
                cleaned_text = clean_text(raw_text)
                
                # Informações de debug
                print(f"Texto extraído do PDF: {len(raw_text)} caracteres")
                print(f"Texto limpo: {len(cleaned_text)} caracteres")
                
                # Cria informações sobre o documento
                doc_info = extract_document_info(doc_name, cleaned_text)
                print(f"Metadados do documento: {doc_info}")
                
                # Divide o texto em chunks - AQUI ESTÁ O PROBLEMA POTENCIAL
                # Pode estar retornando strings em vez de dicionários
                chunks = split_text(cleaned_text, doc_info)
                print(f"Documento dividido em {len(chunks)} chunks.")
                
                # Verifica o tipo de cada chunk para debug
                for i, chunk in enumerate(chunks[:2]):  # Mostra apenas os 2 primeiros para não sobrecarregar o log
                    print(f"Chunk {i} tipo: {type(chunk)}")
                    if isinstance(chunk, dict):
                        print(f"Chunk {i} keys: {chunk.keys()}")
                    else:
                        print(f"Chunk {i} não é um dicionário, é {type(chunk)}")
                
                # Processa os chunks com segurança
                for chunk in chunks:
                    if isinstance(chunk, dict) and "text" in chunk and "metadata" in chunk:
                        # Se for um dicionário com a estrutura esperada
                        all_texts.append(chunk["text"])
                        all_metadatas.append(filter_complex_metadata(chunk["metadata"]))
                    elif isinstance(chunk, str):
                        # Se for uma string
                        all_texts.append(chunk)
                        all_metadatas.append({"source": doc_name})
                    else:
                        # Tipo inesperado, trata como uma string vazia
                        print(f"AVISO: Chunk de tipo inesperado: {type(chunk)}")
                        all_texts.append("")
                        all_metadatas.append({"source": doc_name, "error": "tipo_inesperado"})
            else:
                print(f"AVISO: Arquivo não encontrado: {doc_path}")
        except Exception as e:
            print(f"Erro ao processar {doc_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            print(f"Continuando com os próximos documentos...")
    
    # Cria o banco de dados com todos os textos coletados
    print(f"\nCriando banco de dados com {len(all_texts)} chunks no total...")
    
    db = Chroma.from_texts(
        texts=all_texts,
        embedding=embeddings,
        metadatas=all_metadatas,
        persist_directory=DB_PATH
    )
    
    print(f"Banco de dados vetorial criado com sucesso em '{DB_PATH}'.")
    
    return db

def filter_metadata_dict(metadata_dict):
    """
    Filtra um dicionário de metadados, removendo valores complexos.
    
    Args:
        metadata_dict (dict): Dicionário de metadados
        
    Returns:
        dict: Dicionário filtrado apenas com tipos simples
    """
    filtered = {}
    
    for key, value in metadata_dict.items():
        # Mantém apenas tipos simples: str, int, float, bool
        if isinstance(value, (str, int, float, bool)):
            filtered[key] = value
        # Converte listas para strings separadas por vírgula
        elif isinstance(value, list):
            filtered[key] = ", ".join(str(item) for item in value)
        # Ignora tipos complexos
        else:
            filtered[key] = str(value)
    
    return filtered