# Sistema Preditivo de Obesidade - Tech Challenge Fase 04

Este projeto é uma solução completa de Data Analytics e Machine Learning capaz de prever a classificação de obesidade de uma pessoa com base em seus hábitos comportamentais, estilo de vida e fatores genéticos. 

A solução conta com uma análise exploratória de dados (EDA), modelos preditivos treinados e uma aplicação interativa desenvolvida em **Streamlit** para visualização de dashboards e realização de predições em tempo real.

---

## 👥 Integrantes do Grupo
* **Jaqueline de Souza Oliveira** - RM367545
* **Guilherme Paulo Ferreira** - RM367422
* **Debora Ribeiro de Souza** - RM368669

---

## 🚀 Tecnologias Utilizadas

* **Linguagem**: Python 3.10+
* **Modelagem e Análise**: Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, Joblib
* **Visualização de Dados**: Plotly Express, Plotly Graph Objects, Seaborn, Matplotlib
* **Interface do Usuário**: Streamlit

---

## 📁 Estrutura do Projeto

```text
├── app/
│   └── app.py                        # Aplicação interativa em Streamlit (Dashboard + Formulário + Sobre)
│
├── data/
│   └── Obesity.csv                   # Dataset original com os registros comportamentais e físicos
│
├── models/
│   ├── modelo_obesidade.joblib        # Pipeline completo com o modelo campeão (Random Forest)
│   └── label_encoder_obesidade.joblib # Mapeamento das classes de destino (variável target)
│
├── notebooks/
│   ├── 01_EDA.ipynb                  # Análise Exploratória de Dados completa em português
│   └── 02_Modelo.ipynb               # Treinamento, validação e exportação do modelo de ML
│
├── requirements.txt                  # Dependências do ambiente Python
└── README.md                         # Documentação do projeto
```

---

## 🛡️ Prevenção de Data Leakage (Vazamento de Dados)

Durante a fase de análise exploratória, identificou-se que as variáveis de **Peso** (`Weight`) e **Altura** (`Height`) causavam vazamento de dados (*data leakage*), pois a classificação de obesidade é baseada diretamente na fórmula mecânica do IMC. Se mantidas, o modelo aprenderia apenas uma regra matemática simples de peso e altura, ignorando os hábitos de vida e de saúde.

* **Decisão de Projeto**: As variáveis `Height` (Altura) e `Weight` (Peso) foram **removidas** do treinamento do modelo de Machine Learning.
* **Foco do Modelo**: O classificador foi treinado para aprender interações complexas baseadas em comportamento alimentar, prática esportiva e fatores genéticos (14 variáveis de entrada).
* **Uso da Altura e Peso na Aplicação**: O peso e altura informados no formulário do aplicativo servem apenas para calcular e exibir o IMC real do usuário de forma informativa, acompanhado de dicas personalizadas de saúde baseadas nesse IMC.

---

## 🧠 Modelagem e Resultados

Foram testados e avaliados quatro algoritmos de classificação tradicionais no conjunto de teste. O desempenho em acurácia foi o seguinte:

| Posição | Modelo de Machine Learning | Acurácia (Dados de Teste) | Status |
| :---: | :--- | :---: | :---: |
| **1º** | **Random Forest** | **85,34%** | **Campeão (Embarcado)** 🏆 |
| 2º | Gradient Boosting | 80,85% | Avaliado |
| 3º | Árvore de Decisão | 74,23% | Avaliado |
| 4º | Regressão Logística | 62,41% | Avaliado |

Os artefatos finais foram exportados utilizando o `joblib` em formato de pipeline integrado (pré-processamento + classificador), garantindo que os dados de entrada na aplicação sejam tratados da mesma forma que os dados de treino.

---

## 🖥️ A Aplicação Streamlit

A interface interativa do Streamlit está organizada em 3 abas principais:

1. **📊 Dashboard Analítico**:
   * Apresenta indicadores consolidados (*Big Numbers*), como total de registros, média de idade e média do IMC.
   * Gráficos interativos em Plotly sobre a distribuição de obesidade, relação do IMC com a categoria, meios de transporte preferidos, análise de sedentarismo (atividade física vs uso de telas) e impacto do histórico familiar.
2. **🔮 Predição**:
   * Um formulário amigável 100% em português para preenchimento de hábitos e dados comportamentais.
   * Utiliza controles deslizantes (*sliders*) para entradas numéricas intuitivas.
   * Retorna a predição da categoria de peso estimada pelo modelo e exibe o IMC calculado com dicas personalizadas de saúde baseadas no resultado.
3. **ℹ️ Sobre o Projeto**:
   * Descritivo técnico com os integrantes do grupo, a explicação de prevenção de data leakage e a tabela de modelos testados.

---

## 🛠️ Como Executar Localmente

### 1. Preparar o Ambiente

Certifique-se de ter o Python instalado. Crie um ambiente virtual e ative-o:

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar no Windows (PowerShell/CMD)
venv\Scripts\activate

# Ativar no Linux/macOS
source venv/bin/activate
```

### 2. Instalar as Dependências

Instale os pacotes listados no `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Executar o Streamlit

Inicie a aplicação local:

```bash
streamlit run app/app.py
```

Acesse o endereço `http://localhost:8501` em seu navegador para explorar o dashboard e realizar predições.

---

### 🐳 4. Executar via Docker

Para rodar a aplicação em um container Docker isolado, execute:

```bash
# Construir a imagem
docker build -t previsao-obesidade-app .

# Iniciar o container mapeando a porta 8501
docker run -p 8501:8501 previsao-obesidade-app
```

---

### 🟢 5. Executar via Conda (Alternativa)

Caso prefira gerenciar o ambiente com Anaconda/Miniconda:

```bash
# Criar o ambiente com base no arquivo environment.yml
conda env create -f environment.yml

# Ativar o ambiente
conda activate obesity-prediction

# Rodar o streamlit
streamlit run app/app.py
```

---

### 🛠️ 6. Atalhos com o Makefile

Se você estiver em um ambiente Unix/macOS (ou Windows com Make configurado), você pode usar os atalhos abaixo:

* `make install`: Instala os pacotes necessários.
* `make run`: Executa o app Streamlit localmente.
* `make docker-build`: Constrói a imagem Docker.
* `make docker-run`: Executa o container Docker.
* `make clean`: Limpa caches do python e jupyter.

