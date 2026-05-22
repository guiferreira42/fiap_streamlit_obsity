.PHONY: install run format test docker-build docker-run clean help

# Variáveis
IMAGE_NAME = previsao-obesidade-app
PORT = 8501

help:
	@echo "Comandos disponíveis:"
	@echo "  make install       Instala as dependências do requirements.txt"
	@echo "  make run           Inicia a aplicação Streamlit localmente"
	@echo "  make format        Formata o código usando 'black' (se instalado)"
	@echo "  make test          Executa os testes unitários com 'pytest' (se configurado)"
	@echo "  make docker-build  Gera a imagem Docker do container"
	@echo "  make docker-run    Executa a imagem Docker na porta $(PORT)"
	@echo "  make clean         Remove arquivos temporários e caches"

install:
	pip install -r requirements.txt

run:
	streamlit run app/app.py

format:
	black app/ notebooks/

test:
	pytest

docker-build:
	docker build -t $(IMAGE_NAME) .

docker-run:
	docker run -p $(PORT):$(PORT) $(IMAGE_NAME)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
