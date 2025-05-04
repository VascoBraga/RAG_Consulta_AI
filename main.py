import argparse
from init import load_environment
from bot import run_bot
from knowledge import integrate_consumer_law_documents
from init import diagnose_paths

load_environment()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Sistema RAG de Direito do Consumidor')
    parser.add_argument('--update-only', action='store_true', 
                        help='Apenas atualiza a base de conhecimento sem iniciar o bot')
    parser.add_argument('--run-only', action='store_true',
                        help='Apenas inicia o bot sem atualizar a base de conhecimento')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    load_environment()
    diagnose_paths()

    # Se não for apenas execução, atualiza a base de conhecimento
    if not args.run_only:
        print("Iniciando atualização da base de conhecimento de Direito do Consumidor...")
        db = integrate_consumer_law_documents()
        print("Base de conhecimento atualizada com sucesso!")
        try:
            print(f"Total de documentos na base: {db._collection.count()}")
        except:
            print("Base de conhecimento atualizada.")
    
    # Se não for apenas atualização, executa o bot
    if not args.update_only:
        print("\nIniciando o assistente de Direito do Consumidor...")
        run_bot()
