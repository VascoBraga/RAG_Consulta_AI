import os
from dotenv import load_dotenv
import fitz

def load_environment():
    """Carrega variáveis de ambiente e verifica a API key."""
    load_dotenv(dotenv_path=None, override=True)
    
    # Verifica se a API key está configurada
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("A API key da OpenAI não está configurada. Adicione-a ao arquivo .env")

    #print(f"API KEY carregada: {'*' * 4}{os.getenv('OPENAI_API_KEY')[-4:] if os.getenv('OPENAI_API_KEY') else 'Não Encontrada'}")
    print(f"API KEY carregada: {'*' * 4}{os.getenv('OPENAI_API_KEY')[-4:] if os.getenv('OPENAI_API_KEY') else 'Não Encontrada'}")


# Configurações globais
EMBED_MODEL = "text-embedding-3-small"  # Modelo de embeddings mais recente
LLM_MODEL = "gpt-3.5-turbo"             # Modelo LLM
CHUNK_SIZE = 750                        # Tamanho dos chunks de texto otimizado para textos jurídicos
CHUNK_OVERLAP = 150                     # Sobreposição entre chunks

# Diretórios e caminhos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Diretório base do projeto
LEGISLACAO_DIR = os.path.join(BASE_DIR, "legislacao")  # Diretório com as legislações
DB_PATH = os.path.join(BASE_DIR, "chroma_db")          # Diretório para o banco de dados Chroma

# Em init.py
import os

# Obter o diretório do script atual (src)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Subir um nível para obter o diretório raiz do projeto
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# Definir o caminho para a pasta legislacao
LEGISLACAO_DIR = os.path.join(BASE_DIR, "legislacao")

print(f"BASE_DIR: {BASE_DIR}")
print(f"LEGISLACAO_DIR: {LEGISLACAO_DIR}")

def list_available_files():
    """Lista todos os arquivos disponíveis na pasta de legislação."""
    print("Arquivos disponíveis na pasta legislacao:")
    for filename in os.listdir(LEGISLACAO_DIR):
        file_path = os.path.join(LEGISLACAO_DIR, filename)
        if os.path.isfile(file_path):
            print(f"  - {filename} ({os.path.getsize(file_path)} bytes)")

# Certifica que o diretório de legislações existe
if not os.path.exists(LEGISLACAO_DIR):
    os.makedirs(LEGISLACAO_DIR)
    print(f"Diretório de legislações criado: {LEGISLACAO_DIR}")

# Certifica que o diretório do banco de dados existe
if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)
    print(f"Diretório do banco de dados criado: {DB_PATH}")


# Mapeamento de documentos
def build_document_paths():
    """Constrói o dicionário de documentos com base nos arquivos disponíveis na pasta de legislação."""
    documento_paths = {}
    
    # Mapeamento de nomes de arquivo para títulos descritivos
    filename_to_title = {
        "CDC_2024.pdf": "Código do Consumidor (Lei nº 8.078/90)",
        "lc_166_2019.pdf": "Lei Complementar n.166/2019",
        "lei_13828_2019.pdf": "Lei n.13.828/2019",
        "Lei-9870.pdf": "Lei n. 9.870/1999",
        "Lei-10962.pdf": "Lei n. 10.962/2004",
        "Lei-12741.pdf": "Lei n. 12.741/2012",
        "Lei-12291.pdf": "Lei n. 12.291/2010",
        "D5903.pdf": "Decreto n. 5.903/2006",
        "D7962.pdf": "Decreto n. 7.962/2013",
        "decreto_6523_2008.pdf": "Decreto n. 6.523/2008",
        "D2181.pdf": "Decreto n. 2.181/1997",
        "CNSP_434.pdf": "Resolução CNSP n. 434",
        "CNSP_296.pdf": "Resolução CNSP n. 296"
    }
    
    # Verifica os arquivos disponíveis na pasta
    for filename in os.listdir(LEGISLACAO_DIR):
        filepath = os.path.join(LEGISLACAO_DIR, filename)
        # Verificar se é um arquivo e tem tamanho maior que zero
        if os.path.isfile(filepath) and os.path.getsize(filepath) > 0:
            # Se o arquivo está em nosso mapeamento, usamos o título definido
            if filename in filename_to_title:
                documento_paths[filename_to_title[filename]] = filepath
            else:
                # Para arquivos não mapeados, usamos o nome do arquivo como título
                documento_paths[filename] = filepath
    
    return documento_paths

# Verifica os documentos disponíveis e inclui na base
DOCUMENTO_PATHS = build_document_paths()

def extract_text_from_pdf(pdf_path):
    """Extrai texto do arquivo PDF."""
    # Verificação explícita
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {pdf_path}")
        
    # Imprime informações para debug
    print(f"Abrindo arquivo: {pdf_path}")
    print(f"Arquivo existe: {os.path.exists(pdf_path)}")
    print(f"Tamanho do arquivo: {os.path.getsize(pdf_path)} bytes")
    
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Erro ao abrir o PDF {pdf_path}: {str(e)}")
        print(f"Tipo do erro: {type(e)}")
        raise

def diagnose_paths():
    """Diagnostica problemas com caminhos de arquivos."""
    print("\n=== Diagnóstico de Caminhos ===")
    print(f"Diretório atual: {os.getcwd()}")
    print(f"LEGISLACAO_DIR: {LEGISLACAO_DIR}")
    
    # Verifica se o diretório de legislação existe
    if os.path.exists(LEGISLACAO_DIR):
        print(f"Diretório de legislação existe")
        print("Arquivos no diretório:")
        for filename in os.listdir(LEGISLACAO_DIR):
            file_path = os.path.join(LEGISLACAO_DIR, filename)
            file_size = os.path.getsize(file_path) if os.path.isfile(file_path) else "N/A"
            print(f"  - {filename} ({file_size} bytes)")
    else:
        print(f"ERRO: Diretório de legislação não existe")
        
    print("=== Fim do Diagnóstico ===\n")