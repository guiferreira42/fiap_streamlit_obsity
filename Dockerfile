# Usar imagem oficial do Python como base
FROM python:3.10-slim

# Definir o diretório de trabalho dentro do container
WORKDIR /app

# Instalar dependências de sistema recomendadas
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar apenas o requirements.txt para aproveitar o cache de camadas do Docker
COPY requirements.txt .

# Instalar dependências do Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo o código do projeto para o diretório de trabalho
COPY . .

# Expor a porta padrão utilizada pelo Streamlit
EXPOSE 8501

# Configurações de saúde (Healthcheck) para o container
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Comando de inicialização do aplicativo Streamlit
ENTRYPOINT ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
