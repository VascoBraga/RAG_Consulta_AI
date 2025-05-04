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
    
    # Convertemos para o formato esperado com metadados
    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        # Criamos um dicionário para cada chunk
        chunk = {
            "text": chunk_text,
            "metadata": {
                "chunk_index": i,
                "total_chunks": len(raw_chunks)
            }
        }
        
        # Adicionamos metadados do documento, se fornecidos
        if doc_info:
            chunk["metadata"].update(doc_info)
            
        chunks.append(chunk)
    
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
    #db.persist()
    return db

    db = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    persist_directory="./chroma_db"
)

    #db.persist()


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
    
    Args:
        document_paths (dict): Dicionário com o formato {nome_documento: caminho_arquivo}
        vector_db (Chroma, optional): Instância do banco de dados. Se None, será carregado.
        document_metadata (dict, optional): Metadados adicionais para cada documento.
    
    Returns:
        Chroma: Instância do banco de dados atualizado.
    """
    # Renomeei 'db' para 'vector_db' para evitar colisões de nome
    if vector_db is None:
        vector_db = get_or_create_db()
    
    # Verificação de tipo para garantir que vector_db é um objeto com o método add_texts
    if not hasattr(vector_db, 'add_texts'):
        print(f"AVISO: O objeto vector_db não tem o método 'add_texts'. Tipo: {type(vector_db)}")
        print("Criando um novo banco de dados vetorial...")
        embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
        vector_db = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embeddings
        )
    
    if document_metadata is None:
        document_metadata = {}
    
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    
    total_chunks = 0
    
    for doc_name, doc_path in document_paths.items():
        if not os.path.exists(doc_path):
            print(f"AVISO: Arquivo não encontrado: {doc_path}. Pulando...")
            continue
            
        print(f"Processando documento: {doc_name} ({doc_path})")
        
        try:
            # Detecta o tipo de arquivo
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
            
            # Extrai informação estrutural (número da lei, tipo de documento, etc.)
            doc_info = extract_document_info(doc_name, cleaned_text)
            
            # Adiciona metadados personalizados, se disponíveis
            if doc_name in document_metadata:
                doc_info.update(document_metadata[doc_name])
            
            # Divide o texto considerando a estrutura de documentos jurídicos
            chunks = split_legal_text(cleaned_text, doc_info)
            
            print(f"  -> Documento dividido em {len(chunks)} chunks.")
            total_chunks += len(chunks)
            
            # Adiciona os novos chunks ao banco de dados
            texts = []
            metadatas = []
            
            for chunk in chunks:
                if isinstance(chunk, dict) and "text" in chunk and "metadata" in chunk:
					# Formato correto
                    texts.append(chunk["text"])
					# Filtra metadados complexos
                    metadatas.append(chunk["metadata"])
                elif isinstance(chunk, str):
					# Se for uma string, criar metadados básicos
                    texts.append(chunk)
                    metadata = {"source": doc_name, "chunk_index": len(texts) - 1}
					# Adiciona metadados do documento se disponíveis
                    if doc_name in document_metadata:
                        metadata.update(document_metadata[doc_name])
                    metadatas.append(metadata)

			# Agora filtra os metadados após garantir que todos estão no formato correto
            filtered_metadatas = [filter_complex_metadata(m) for m in metadatas]
						
            try:
                vector_db.add_texts(texts=texts, metadatas=filtered_metadatas)
                print(f"  -> Chunks adicionados ao banco de dados com sucesso.")
            except TypeError as e:
                if "unexpected keyword argument 'embeddings'" in str(e):
                    # Algumas versões do Chroma não aceitam o parâmetro embeddings diretamente
                    vector_db.add_texts(texts=texts, metadatas=filtered_metadatas)
                    print(f"  -> Chunks adicionados ao banco de dados com sucesso (sem embeddings explícitos).")
                else:
                    raise
                    
        except Exception as e:
            print(f"Erro ao processar documento {doc_name}: {str(e)}")
            print(f"Continuando com os próximos documentos...")
    
    # Persiste as mudanças
    try:
        #vector_db.persist()
        print(f"Banco de dados atualizado com sucesso. Total de {total_chunks} novos chunks adicionados.")
    except Exception as e:
        print(f"AVISO: Não foi possível persistir o banco de dados: {str(e)}")
    
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
    
    return validated_chunks


def integrate_consumer_law_documents():
    """Integra múltiplos documentos de legislação do consumidor na base de conhecimento."""
    # Dicionário de documentos a serem processados
    documents = {
        "Código do Consumidor (Lei 8.078/90)": "CDC_2024.pdf",
        "Decreto 11.034/2022": "decreto_11034_2022.pdf",
        "Lei 14.034/2020": "lei_14034_2020.pdf",
        "Lei Complementar 166/2019": "lc_166_2019.pdf",
        "Lei 13.828/2019": "lei_13828_2019.pdf",
        "Lei 10.962/2004": "lei_10962_2004.pdf",
        "Decreto 5.903/2006": "decreto_5903_2006.pdf",
        "Lei 12.741/2012": "lei_12741_2012.pdf",
        "Lei 12.291/2010": "lei_12291_2010.pdf",
        "Decreto 7.962/2013": "decreto_7962_2013.pdf",
        "Decreto 6.523/2008": "decreto_6523_2008.pdf",
        "Decreto 2.181/1997": "decreto_2181_1997.pdf",
        "Resolução CNSP 107/2004": "resolucao_cnsp_107_2004.pdf",
        "Resolução CNSP 296": "resolucao_cnsp_296.pdf"
    }
    
    # Metadados adicionais para cada documento
    metadata = {
        "Código do Consumidor (Lei 8.078/90)": {
            "description": "Código de Defesa do Consumidor - Lei principal que estabelece normas de proteção e defesa do consumidor",
            "importance": "alta",
            "category": "direitos_basicos"
        },
        "Decreto 11.034/2022": {
            "description": "Regulamenta o atendimento via SAC",
            "importance": "média", 
            "category": "atendimento"
        },
        # Adicionar metadados para os outros documentos...
    
        "Código do Consumidor (Lei nº 8.078/90)": {
            "description": "Lei principal que estabelece normas de proteção e defesa do consumidor nas relações de consumo",
            "importance": "alta",
            "category": "direitos_basicos",
            "publication_date": "11/09/1990",
            "keywords": ["defesa do consumidor", "direitos básicos", "práticas abusivas", "produtos e serviços", "responsabilidade civil"],
            "summary": "Estabelece direitos básicos dos consumidores, regras de qualidade de produtos e serviços, práticas comerciais, proteção contratual e sanções administrativas",
            "scope": "geral",
            "hierarchy": "lei_principal"
        },
        
        "Lei Complementar n.166/2019": {
            "description": "Dispõe sobre cadastros positivos de crédito e regula a anotação de informações de adimplemento",
            "importance": "média",
            "category": "credito_financeiro",
            "publication_date": "08/04/2019",
            "keywords": ["cadastro positivo", "proteção de dados", "score de crédito", "bancos de dados"],
            "summary": "Regulamenta o cadastro positivo de crédito, permitindo o compartilhamento de informações sobre adimplemento entre gestores de bancos de dados",
            "scope": "especifico",
            "hierarchy": "lei_complementar"
        },
        
        "Lei n.13.828/2019": {
            "description": "Altera a Lei nº 8.078/90 (CDC) para autorizar a autoridade administrativa a requisitar auxílio de força policial",
            "importance": "média",
            "category": "fiscalizacao",
            "publication_date": "13/05/2019",
            "keywords": ["fiscalização", "força policial", "autoridade administrativa", "poder de polícia"],
            "summary": "Permite que autoridades administrativas de defesa do consumidor requisitem auxílio de força policial para cumprimento de suas determinações",
            "scope": "especifico",
            "hierarchy": "lei_ordinaria",
            "modifies": "Lei nº 8.078/90"
        },
        
        "Lei n. 10.962/2004": {
            "description": "Dispõe sobre a oferta e as formas de afixação de preços de produtos e serviços para o consumidor",
            "importance": "média",
            "category": "informacao_preco",
            "publication_date": "11/10/2004",
            "keywords": ["afixação de preços", "informação", "etiquetas", "código de barras"],
            "summary": "Regula a oferta e as formas de afixação de preços em vitrines e a marcação em produtos expostos a venda ao consumidor",
            "scope": "especifico",
            "hierarchy": "lei_ordinaria"
        },
        
        "Decreto n. 5.903/2006": {
            "description": "Regulamenta a Lei n. 10.962/2004 e dispõe sobre as práticas infracionais na oferta de produtos e serviços",
            "importance": "média",
            "category": "informacao_preco",
            "publication_date": "20/09/2006",
            "keywords": ["afixação de preços", "código de barras", "precificação", "infrações"],
            "summary": "Detalha as regras para afixação de preços, uso de código de barras e estabelece as práticas consideradas infracionais",
            "scope": "especifico",
            "hierarchy": "decreto_regulamentador",
            "regulates": "Lei n. 10.962/2004"
        },
        
        "Lei n. 12.741/2012": {
            "description": "Dispõe sobre medidas de clareza na informação sobre tributos incidentes sobre mercadorias e serviços",
            "importance": "média",
            "category": "informacao_tributaria",
            "publication_date": "08/12/2012",
            "keywords": ["tributos", "carga tributária", "nota fiscal", "informação ao consumidor"],
            "summary": "Obriga a informação do valor aproximado dos tributos incidentes sobre mercadorias e serviços nos documentos fiscais",
            "scope": "especifico",
            "hierarchy": "lei_ordinaria"
        },
        
        "Lei n. 12.291/2010": {
            "description": "Torna obrigatória a manutenção de exemplar do Código de Defesa do Consumidor nos estabelecimentos comerciais",
            "importance": "baixa",
            "category": "informacao_consumidor",
            "publication_date": "20/07/2010",
            "keywords": ["disponibilização do CDC", "estabelecimentos comerciais", "informação"],
            "summary": "Obriga estabelecimentos comerciais a manterem exemplar do CDC disponível para consulta do público em local visível",
            "scope": "especifico",
            "hierarchy": "lei_ordinaria"
        },
        
        "Decreto n. 7.962/2013": {
            "description": "Regulamenta o CDC para dispor sobre a contratação no comércio eletrônico",
            "importance": "alta",
            "category": "comercio_eletronico",
            "publication_date": "15/03/2013",
            "keywords": ["comércio eletrônico", "e-commerce", "contratação à distância", "internet"],
            "summary": "Estabelece regras para compras online, incluindo identificação do fornecedor, apresentação de informações e direito de arrependimento",
            "scope": "especifico",
            "hierarchy": "decreto_regulamentador",
            "regulates": "Lei nº 8.078/90"
        },
        
        "Decreto n. 6.523/2008": {
            "description": "Regulamenta o CDC para fixar normas sobre o Serviço de Atendimento ao Consumidor (SAC)",
            "importance": "alta",
            "category": "atendimento",
            "publication_date": "31/07/2008",
            "keywords": ["SAC", "call center", "atendimento telefônico", "reclamações"],
            "summary": "Estabelece regras para o funcionamento dos SACs telefônicos, incluindo tempo de espera, opções de atendimento e resolução de demandas",
            "scope": "especifico",
            "hierarchy": "decreto_regulamentador",
            "regulates": "Lei nº 8.078/90"
        },
        
        "Decreto n. 2.181/1997": {
            "description": "Dispõe sobre a organização do Sistema Nacional de Defesa do Consumidor (SNDC) e estabelece normas gerais sobre sanções administrativas",
            "importance": "alta",
            "category": "fiscalizacao_sancoes",
            "publication_date": "20/03/1997",
            "keywords": ["SNDC", "fiscalização", "sanções administrativas", "processo administrativo"],
            "summary": "Organiza o SNDC e estabelece os procedimentos administrativos para aplicação de sanções por violação ao CDC",
            "scope": "geral",
            "hierarchy": "decreto_regulamentador",
            "regulates": "Lei nº 8.078/90"
        },
        
        "Resolução CNSP n. 107/2004": {
            "description": "Estabelece regras e critérios para os planos de seguro sob a forma de bilhete",
            "importance": "baixa",
            "category": "seguros",
            "publication_date": "16/01/2004",
            "keywords": ["seguro", "bilhete de seguro", "direitos do segurado", "contratos de adesão"],
            "summary": "Regula os planos de seguro sob a forma de bilhete, com transparência e proteção ao consumidor",
            "scope": "especifico",
            "hierarchy": "resolucao_normativa",
            "regulatory_agency": "CNSP"
        },
        
        "Resolução CNSP n. 296": {
            "description": "Institui regras e procedimentos para intermediação de seguros e resseguros",
            "importance": "baixa",
            "category": "seguros",
            "publication_date": "25/10/2013",
            "keywords": ["seguro", "intermediação", "corretor", "transparência"],
            "summary": "Estabelece regras de conduta, transparência e proteção ao consumidor na intermediação de contratos de seguro",
            "scope": "especifico",
            "hierarchy": "resolucao_normativa",
            "regulatory_agency": "CNSP"
        }
    }

    
    # Carrega o banco de dados existente
    vector_db = get_or_create_db()
    
    # Atualiza o banco com os novos documentos
    vector_db = update_vector_db(DOCUMENTO_PATHS, vector_db, metadata)
    
    return vector_db

def configure_advanced_retriever(db):
    """Configura um retriever avançado que aproveita os metadados dos documentos."""
    
    # Retriever base com MMR para maior diversidade
    base_retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 10,
            "lambda_mult": 0.7
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
                
                # Cria informações sobre o documento
                doc_info = extract_document_info(doc_name, cleaned_text)
                
                # Divide o texto em chunks
                chunks = split_legal_text(cleaned_text, doc_info)
                print(f"Documento dividido em {len(chunks)} chunks.")
                
                # Adiciona os chunks ao banco
                for chunk in chunks:
                    if isinstance(chunk, dict) and "text" in chunk and "metadata" in chunk:
                        all_texts.append(chunk["text"])
                        all_metadatas.append(filter_complex_metadata(chunk["metadata"]))
                    elif isinstance(chunk, str):
                        all_texts.append(chunk)
                        metadata = {"source": doc_name}
                        all_metadatas.append(metadata)
            else:
                print(f"AVISO: Arquivo não encontrado: {doc_path}")
        except Exception as e:
            print(f"Erro ao processar {doc_name}: {str(e)}")
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