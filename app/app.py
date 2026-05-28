import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import numpy as np

# Configuração da página
st.set_page_config(page_title="Previsão de Obesidade", page_icon="📊", layout="wide")

# Caminhos absolutos/relativos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Obesity.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'modelo_obesidade.joblib')
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'label_encoder_obesidade.joblib')

@st.cache_data
def carregar_dados():
    try:
        df = pd.read_csv(DATA_PATH)
        df['BMI'] = df['Weight'] / (df['Height'] ** 2)
        return df
    except FileNotFoundError:
        st.error(f"Arquivo não encontrado em: {DATA_PATH}")
        return pd.DataFrame()

@st.cache_resource
def carregar_artefatos():
    try:
        modelo = joblib.load(MODEL_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        return modelo, label_encoder
    except Exception as e:
        st.error(f"Erro ao carregar os modelos: {e}. Por favor, execute o notebook de modelagem primeiro.")
        return None, None

df = carregar_dados()
modelo, label_encoder = carregar_artefatos()

# Layout
st.title("Sistema Preditivo de Obesidade")
st.markdown("Projeto de Machine Learning - Tech Challenge Fase 04")

tab1, tab2, tab3 = st.tabs(["📊 Dashboard Analítico", "🔮 Predição", "ℹ️ Sobre o Projeto"])

with tab1:
    st.header("Dashboard Analítico")
    if not df.empty:
        df_translated = df.copy()
        
        translation_dict = {
            'Insufficient_Weight': 'Abaixo do Peso',
            'Normal_Weight': 'Peso Normal',
            'Overweight_Level_I': 'Sobrepeso Grau I',
            'Overweight_Level_II': 'Sobrepeso Grau II',
            'Obesity_Type_I': 'Obesidade Grau I',
            'Obesity_Type_II': 'Obesidade Grau II',
            'Obesity_Type_III': 'Obesidade Grau III'
        }
        
        if 'Obesity' in df_translated.columns:
            df_translated['Obesity'] = df_translated['Obesity'].map(translation_dict).fillna(df_translated['Obesity'])
            
        if 'Gender' in df_translated.columns:
            df_translated['Gender'] = df_translated['Gender'].map({
                'Female': 'Feminino',
                'Male': 'Masculino'
            }).fillna(df_translated['Gender'])
            
        if 'family_history' in df_translated.columns:
            df_translated['family_history'] = df_translated['family_history'].map({
                'yes': 'Sim',
                'no': 'Não'
            }).fillna(df_translated['family_history'])
            
        if 'MTRANS' in df_translated.columns:
            df_translated['MTRANS'] = df_translated['MTRANS'].map({
                'Public_Transportation': 'Transporte Público',
                'Walking': 'Caminhada',
                'Automobile': 'Automóvel',
                'Motorbike': 'Motocicleta',
                'Bike': 'Bicicleta'
            }).fillna(df_translated['MTRANS'])

        category_order = ['Abaixo do Peso', 'Peso Normal', 'Sobrepeso Grau I', 'Sobrepeso Grau II', 'Obesidade Grau I', 'Obesidade Grau II', 'Obesidade Grau III']

        # Métricas (Big Numbers)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total de Registros", f"{df.shape[0]:,}".replace(",", "."))
        col2.metric("Total de Variáveis (Features)", df.shape[1] - 1) 
        col3.metric("Média de Idade", f"{df['Age'].mean():.1f} anos")
        col4.metric("Média de IMC (Calculado)", f"{df['BMI'].mean():.1f}")
        
        st.markdown("---")
        
        # Primeira linha de gráficos
        col_fig1, col_fig2 = st.columns(2)
        
        with col_fig1:
            fig1 = px.histogram(df_translated, y="Obesity", color="Obesity", 
                                category_orders={"Obesity": category_order},
                                labels={"Obesity": "Nível de Obesidade", "count": "Quantidade"},
                                title="Distribuição dos Níveis de Obesidade")
            fig1.update_layout(showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)
            
        with col_fig2:
            fig2 = px.box(df_translated, x="BMI", y="Obesity", color="Obesity",
                          category_orders={"Obesity": category_order},
                          labels={"BMI": "IMC (Índice de Massa Corporal)", "Obesity": "Nível de Obesidade"},
                          title="IMC vs Nível de Obesidade")
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
            
        st.markdown("---")
        
        # Segunda linha de gráficos (Transporte e Sedentarismo)
        col_fig3, col_fig4 = st.columns(2)
        
        with col_fig3:
            fig_trans = px.histogram(df_translated, x="MTRANS", color="Obesity",
                                     category_orders={"Obesity": category_order},
                                     labels={"MTRANS": "Meio de Transporte", "count": "Quantidade", "Obesity": "Nível de Obesidade"},
                                     title="Meio de Transporte Utilizado por Nível de Obesidade")
            fig_trans.update_layout(legend_title_text="Nível de Obesidade")
            st.plotly_chart(fig_trans, use_container_width=True)
            
        with col_fig4:
            fig_sed = px.scatter(df_translated, x="FAF", y="TUE", color="Obesity",
                                 category_orders={"Obesity": category_order},
                                 labels={"FAF": "Atividade Física (Dias/Semana)", "TUE": "Uso de Telas (Horas/Dia)", "Obesity": "Nível de Obesidade"},
                                 title="Sedentarismo: Atividade Física vs. Uso de Telas",
                                 opacity=0.7)
            fig_sed.update_layout(legend_title_text="Nível de Obesidade")
            st.plotly_chart(fig_sed, use_container_width=True)
            
        st.markdown("---")
        
        # Terceira linha (Gráfico Sugerido: Histórico Familiar)
        st.subheader("Gráfico Sugerido: Histórico Familiar")
        fig_fam = px.histogram(df_translated, x="family_history", color="Obesity",
                               category_orders={"Obesity": category_order},
                               labels={"family_history": "Histórico Familiar de Sobrepeso", "count": "Quantidade", "Obesity": "Nível de Obesidade"},
                               title="Histórico Familiar vs. Nível de Obesidade",
                               barmode="group")
        fig_fam.update_layout(legend_title_text="Nível de Obesidade")
        st.plotly_chart(fig_fam, use_container_width=True)
        
    else:
        st.warning("Dados não carregados.")

def obter_dicas_imc(imc):
    if imc < 18.5:
        return {
            "categoria": "Abaixo do peso",
            "cor": "warning",
            "dicas": [
                "**Alimentação Nutritiva**: Priorize alimentos ricos em nutrientes e calorias saudáveis (como castanhas, abacate, azeite de oliva e proteínas magras).",
                "**Frequência**: Faça refeições menores e mais frequentes ao longo do dia para facilitar a ingestão calórica.",
                "**Exercício Físico**: Pratique exercícios de resistência/musculação para ajudar no ganho saudável de massa muscular magra.",
                "**Acompanhamento**: Agende uma consulta com um nutricionista para estruturar um plano de ganho de peso saudável."
            ]
        }
    elif imc < 25.0:
        return {
            "categoria": "Peso ideal (Saudável)",
            "cor": "success",
            "dicas": [
                "**Manutenção**: Continue com seus ótimos hábitos de alimentação equilibrada, rica em vegetais, frutas e grãos integrais.",
                "**Atividade Física**: Mantenha uma rotina de exercícios físicos de pelo menos 150 minutos por semana (misturando cárdio e fortalecimento).",
                "**Hidratação**: Beba água regularmente para manter as funções metabólicas em dia.",
                "**Qualidade do Sono**: Mantenha um padrão de sono regular de 7 a 8 horas por noite."
            ]
        }
    elif imc < 30.0:
        return {
            "categoria": "Sobrepeso",
            "cor": "warning",
            "dicas": [
                "**Alimentação**: Reduza o consumo de alimentos ultraprocessados, açúcares e bebidas calóricas. Aumente o consumo de fibras e vegetais.",
                "**Atividade Física**: Tente intensificar seus treinos. Atividades como caminhadas rápidas, corrida ou ciclismo por 4 a 6 dias na semana ajudam a aumentar o gasto calórico.",
                "**Rotina**: Monitore o tempo que passa sentado ou usando telas (procure limitar a menos de 2-3 horas por dia fora do trabalho).",
                "**Orientação profissional**: Um nutricionista ou educador físico pode ajudar a criar metas sustentáveis de perda de peso."
            ]
        }
    else:
        return {
            "categoria": "Obesidade",
            "cor": "danger",
            "dicas": [
                "**Apoio Médico**: A obesidade é uma condição de saúde complexa. Recomendamos buscar orientação de um médico endocrinologista e nutricionista para um plano integrado.",
                "**Exercícios de Baixo Impacto**: Inicie com atividades de baixo impacto para proteger suas articulações, como caminhada leve, hidroginástica ou natação.",
                "**Mudança Gradual**: Evite dietas extremamente restritivas. Foque em pequenas mudanças de longo prazo em seus hábitos alimentares diários.",
                "**Hábitos Saudáveis**: Priorize o aumento no consumo de água, melhore a qualidade do sono e busque gerenciar os níveis de estresse cotidianos."
            ]
        }

with tab2:
    st.header("Predição de Nível de Obesidade")
    st.markdown("Preencha o formulário abaixo com seus hábitos e dados físicos para prever o nível de obesidade.")
    
    if modelo is not None and not df.empty:
        with st.form("form_predicao"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Dados Pessoais")
                gender = st.selectbox("Gênero", ["Feminino", "Masculino"])
                age = st.number_input("Idade", min_value=10, max_value=100, value=25, step=1)
                height = st.number_input("Altura em metros", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
                weight = st.number_input("Peso em kg", min_value=30.0, max_value=200.0, value=70.0, step=1.0)
                family_history = st.selectbox("Histórico Familiar de Sobrepeso", ["Sim", "Não"])
                
            with col2:
                st.subheader("Hábitos Físicos")
                faf_desc = st.selectbox("Frequência semanal de atividade física", ["Nenhuma", "1 a 2 vezes/semana", "3 a 4 vezes/semana", "5 ou mais vezes/semana"], index=1)
                tue_desc = st.selectbox("Tempo diário usando dispositivos eletrônicos", ["0 a 2 horas/dia", "3 a 5 horas/dia", "Mais de 5 horas/dia"], index=1)
                mtrans = st.selectbox("Meio de transporte", ["Transporte Público", "Caminhada", "Automóvel", "Motocicleta", "Bicicleta"])
            
            with col3:
                st.subheader("Hábitos Alimentares")
                favc = st.selectbox("Consome alimentos calóricos frequente", ["Sim", "Não"])
                fcvc = st.slider("Frequência de consumo de vegetais (1-3)", min_value=1, max_value=3, value=2)
                ncp = st.slider("Número de refeições principais (1-4)", min_value=1, max_value=4, value=3)
                caec = st.selectbox("Frequência que se alimenta entre as refeições", ["Não consome", "Às vezes", "Frequentemente", "Sempre"])
                smoke = st.selectbox("Fumante", ["Sim", "Não"])
                ch2o_desc = st.selectbox("Consumo diário de água", ["Menos de 1 L/dia", "1 a 2 L/dia", "Mais de 2 L/dia"], index=1)
                calc = st.selectbox("Consumo de álcool", ["Não consome", "Às vezes", "Frequentemente", "Sempre"])
                
            submit = st.form_submit_button("Realizar Predição")
            
            if submit:
                # Mapeamento dos campos em português para o inglês esperado pelo modelo
                gender_map = {"Feminino": "Female", "Masculino": "Male"}
                yes_no_map = {"Sim": "yes", "Não": "no"}
                caec_map = {
                    "Não consome": "no",
                    "Às vezes": "Sometimes",
                    "Frequentemente": "Frequently",
                    "Sempre": "Always"
                }
                calc_map = {
                    "Não consome": "no",
                    "Às vezes": "Sometimes",
                    "Frequentemente": "Frequently",
                    "Sempre": "Always"
                }
                mtrans_map = {
                    "Transporte Público": "Public_Transportation",
                    "Caminhada": "Walking",
                    "Automóvel": "Automobile",
                    "Motocicleta": "Motorbike",
                    "Bicicleta": "Bike"
                }
                faf_map = {
                    "Nenhuma": 0.0,
                    "1 a 2 vezes/semana": 1.0,
                    "3 a 4 vezes/semana": 2.0,
                    "5 ou mais vezes/semana": 3.0
                }
                tue_map = {
                    "0 a 2 horas/dia": 0.0,
                    "3 a 5 horas/dia": 1.0,
                    "Mais de 5 horas/dia": 2.0
                }
                ch2o_map = {
                    "Menos de 1 L/dia": 1.0,
                    "1 a 2 L/dia": 2.0,
                    "Mais de 2 L/dia": 3.0
                }
                
                # Criar o dataframe com o input mapeado
                input_data = {
                    'Gender': gender_map[gender],
                    'Age': float(age),
                    'family_history': yes_no_map[family_history],
                    'FAVC': yes_no_map[favc],
                    'FCVC': float(fcvc),
                    'NCP': float(ncp),
                    'CAEC': caec_map[caec],
                    'SMOKE': yes_no_map[smoke],
                    'CH2O': ch2o_map[ch2o_desc],
                    'SCC': df['SCC'].mode()[0] if not df.empty and 'SCC' in df.columns else 'no',
                    'FAF': faf_map[faf_desc],
                    'TUE': tue_map[tue_desc],
                    'CALC': calc_map[calc],
                    'MTRANS': mtrans_map[mtrans]
                }
                
                input_df = pd.DataFrame([input_data])
                
                # Calcular IMC para exibição
                bmi = weight / (height ** 2)
                
                # Predição
                try:
                    pred = modelo.predict(input_df)
                    prob = modelo.predict_proba(input_df)
                    
                    pred_class = label_encoder.inverse_transform(pred)[0]
                    max_prob = np.max(prob) * 100
                    
                    # Dicionário de tradução dos níveis de obesidade
                    translation_dict = {
                        'Insufficient_Weight': 'Abaixo do Peso',
                        'Normal_Weight': 'Peso Normal',
                        'Overweight_Level_I': 'Sobrepeso Grau I',
                        'Overweight_Level_II': 'Sobrepeso Grau II',
                        'Obesity_Type_I': 'Obesidade Grau I',
                        'Obesity_Type_II': 'Obesidade Grau II',
                        'Obesity_Type_III': 'Obesidade Grau III'
                    }
                    pred_class_pt = translation_dict.get(pred_class, pred_class)
                    
                    st.markdown("---")
                    st.subheader("🎯 Resultado do Diagnóstico")
                    
                    # Exibir Dicas baseadas no IMC calculado
                    dicas_info = obter_dicas_imc(bmi)
                    
                    res_col1, res_col2, res_col3 = st.columns(3)
                    
                    with res_col1:
                        st.markdown(
                            f"""
                            <div style="
                                background-color: rgba(16, 185, 129, 0.08);
                                border: 1px solid rgba(16, 185, 129, 0.2);
                                border-left: 5px solid #10b981;
                                padding: 15px;
                                border-radius: 8px;
                                min-height: 110px;
                                display: flex;
                                flex-direction: column;
                                justify-content: center;
                            ">
                                <span style="font-size: 13px; color: #a7f3d0; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px;">Classificação do IMC</span>
                                <span style="font-size: 22px; font-weight: bold; color: #10b981; margin-top: 5px; display: block;">{dicas_info['categoria']}</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                    with res_col2:
                        st.markdown(
                            f"""
                            <div style="
                                background-color: rgba(28, 100, 242, 0.08);
                                border: 1px solid rgba(28, 100, 242, 0.2);
                                border-left: 5px solid #1c64f2;
                                padding: 15px;
                                border-radius: 8px;
                                min-height: 110px;
                                display: flex;
                                flex-direction: column;
                                justify-content: center;
                            ">
                                <span style="font-size: 13px; color: #93c5fd; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px;">Resultado Previsto</span>
                                <span style="font-size: 22px; font-weight: bold; color: #3b82f6; margin-top: 5px; display: block;">{pred_class_pt}</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                    with res_col3:
                        st.markdown(
                            f"""
                            <div style="
                                background-color: rgba(28, 100, 242, 0.08);
                                border: 1px solid rgba(28, 100, 242, 0.2);
                                border-left: 5px solid #1c64f2;
                                padding: 15px;
                                border-radius: 8px;
                                min-height: 110px;
                                display: flex;
                                flex-direction: column;
                                justify-content: center;
                            ">
                                <span style="font-size: 13px; color: #93c5fd; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px;">Seu IMC</span>
                                <span style="font-size: 24px; font-weight: bold; color: #3b82f6; margin-top: 5px; display: block;">{bmi:.1f}</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("#### 💡 Dicas de Saúde e Bem-Estar Baseadas no seu IMC:")
                    for dica in dicas_info["dicas"]:
                        st.markdown(f"- {dica}")
                    
                except Exception as e:
                    st.error(f"Ocorreu um erro durante a predição: {e}")
    else:
         st.warning("Modelos não carregados para realizar a predição.")

with tab3:
    st.header("Sobre o Projeto")
    st.markdown("""
    Este projeto foi desenvolvido como parte do **Tech Challenge Fase 04** da Pós-Tech FIAP em Data Analytics.
    
    ### 👥 Integrantes do Grupo
    
    - **Jaqueline de Souza Oliveira** - RM367545
    - **Guilherme Paulo Ferreira** - RM367422
    - **Debora Ribeiro de Souza** - RM368669

    
    ### 🎯 Objetivo do Projeto
    Criar uma solução completa de Data Analytics e Machine Learning capaz de prever o nível de obesidade de uma pessoa com base em suas características físicas, hábitos alimentares, estilo de vida e fatores genéticos.
    
    ### 🛡️ Prevenção de Data Leakage (Vazamento de Dados)
    Durante a análise exploratória dos dados (EDA), identificou-se que as variáveis de **Altura** (`Height`) e **Peso** (`Weight`) possuem relação mecânica direta com as categorias de Obesidade, pois a própria classificação de obesidade é definida pelas faixas de IMC.
    
    Para evitar que o modelo ficasse "viciado" apenas nestas duas variáveis (ignorando todo o resto), **a Altura e o Peso foram removidos** durante o treinamento do modelo. 
    
    Isso forçou os algoritmos a aprenderem os padrões e interações complexas de hábitos comportamentais e genéticos, tornando a ferramenta um verdadeiro **analisador de hábitos de saúde** e não uma mera calculadora matemática de IMC. O Peso e a Altura informados no formulário de predição servem unicamente para fins informativos de exibição de IMC e dicas de saúde na tela.
    
    ### 📊 Modelos e Resultados do Notebook (`02_Modelo.ipynb`)
    Durante a etapa de modelagem, foram testados e comparados quatro algoritmos tradicionais de classificação. Os desempenhos obtidos no conjunto de teste foram:
    
    | Posição | Modelo de Machine Learning | Acurácia (Dados de Teste) |
    | :---: | :--- | :---: |
    | **1º** | **Random Forest** | **85,34%** 🏆 (Selecionado) |
    | 2º | Gradient Boosting | 80,85% |
    | 3º | Árvore de Decisão | 74,23% |
    | 4º | Regressão Logística | 62,41% |
    
    O modelo com melhor acurácia (**Random Forest** com **85,34%**) foi salvo e embarcado nesta aplicação.
    
    ### 🛠️ Pipeline de Produção
    O modelo final foi encapsulado em um `Pipeline` completo utilizando o `scikit-learn` e exportado via `joblib`. Esse pipeline contém:
    1. **Pré-processamento**: Tratamento de valores faltantes, normalização com `StandardScaler` e codificação de categorias com `OneHotEncoder`.
    2. **Classificador**: O algoritmo Random Forest treinado.
    
    Graças a essa arquitetura, a aplicação Streamlit apenas carrega a pipeline integrada e executa a previsão diretamente sobre os dados brutos de entrada, mantendo a consistência e facilidade de manutenção em produção.
    """)
