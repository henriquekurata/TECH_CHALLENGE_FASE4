import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Configuração inicial da página
# -------------------------------
st.set_page_config(
    page_title="Sistema Preditivo de Obesidade",
    layout="wide"
)

st.title("🧠 Sistema Preditivo de Obesidade")
st.markdown(
    """
    Este sistema utiliza **Machine Learning** para auxiliar a equipe médica
    na **predição do nível de obesidade** de pacientes, com base em dados
    físicos e comportamentais.
    
    ⚠️ *Ferramenta de apoio à decisão clínica.*
    """
)

# -------------------------------
# Carregar modelo e dados
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("modelo_obesidade.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("Obesity.csv")

model = load_model()
df = load_data()

# -------------------------------
# Abas do sistema
# -------------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 Visão Analítica",
    "🧠 Sistema Preditivo",
    "📈 Insights do Modelo"
])

# ===============================
# 📊 ABA 1 — VISÃO ANALÍTICA
# ===============================
with tab1:
    st.header("📊 Análise Exploratória dos Dados")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribuição dos níveis de obesidade")
        fig1, ax1 = plt.subplots()
        sns.countplot(
            y=df["Obesity_level"],
            order=df["Obesity_level"].value_counts().index,
            ax=ax1
        )
        st.pyplot(fig1)

    with col2:
        st.subheader("Atividade física x Obesidade")
        fig2, ax2 = plt.subplots()
        sns.boxplot(
            x="Obesity_level",
            y="FAF",
            data=df,
            ax=ax2
        )
        plt.xticks(rotation=45)
        st.pyplot(fig2)

    st.subheader("Relação entre idade, peso e obesidade")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(
        data=df,
        x="Age",
        y="Weight",
        hue="Obesity_level",
        alpha=0.6,
        ax=ax3
    )
    st.pyplot(fig3)

# ===============================
# 🧠 ABA 2 — SISTEMA PREDITIVO
# ===============================
with tab2:
    st.header("🧠 Previsão do Nível de Obesidade")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            Gender = st.selectbox("Gênero", ["Male", "Female"])
            Age = st.slider("Idade", 10, 80, 30)
            Height = st.slider("Altura (m)", 1.40, 2.10, 1.70)
            Weight = st.slider("Peso (kg)", 40.0, 160.0, 70.0)

        with col2:
            family_history = st.selectbox("Histórico familiar de obesidade", ["yes", "no"])
            FAVC = st.selectbox("Consumo frequente de alimentos calóricos", ["yes", "no"])
            FCVC = st.slider("Consumo de vegetais", 1.0, 3.0, 2.0)
            NCP = st.slider("Número de refeições principais", 1.0, 4.0, 3.0)

        with col3:
            CH2O = st.slider("Consumo diário de água", 1.0, 3.0, 2.0)
            FAF = st.slider("Frequência de atividade física", 0.0, 3.0, 1.0)
            TUE = st.slider("Uso de tecnologia (horas/dia)", 0.0, 2.0, 1.0)
            CALC = st.selectbox("Consumo de álcool", ["no", "Sometimes", "Frequently", "Always"])

            CAEC = st.selectbox("Alimentação entre refeições", ["no", "Sometimes", "Frequently", "Always"])
            SMOKE = st.selectbox("Fuma?", ["yes", "no"])
            SCC = st.selectbox("Monitora calorias?", ["yes", "no"])
            MTRANS = st.selectbox(
                "Meio de transporte",
                ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"]
            )

        submitted = st.form_submit_button("🔍 Realizar Análise Preditiva")

    if submitted:
        input_data = pd.DataFrame([{
            "Gender": Gender,
            "Age": Age,
            "Height": Height,
            "Weight": Weight,
            "family_history": family_history,
            "FAVC": FAVC,
            "FCVC": FCVC,
            "NCP": NCP,
            "CAEC": CAEC,
            "SMOKE": SMOKE,
            "CH2O": CH2O,
            "SCC": SCC,
            "FAF": FAF,
            "TUE": TUE,
            "CALC": CALC,
            "MTRANS": MTRANS
        }])

        prediction = model.predict(input_data)[0]

        st.success(f"🧠 Resultado da análise preditiva: **{prediction}**")
        st.info("Este sistema é uma ferramenta de apoio à decisão clínica.")

# ===============================
# 📈 ABA 3 — INSIGHTS DO MODELO
# ===============================
with tab3:
    st.header("📈 Avaliação e Insights do Modelo")

    st.markdown("""
    O modelo selecionado foi o **Gradient Boosting Classifier**, escolhido após
    a comparação com outros algoritmos de Machine Learning.
    """)

    st.markdown("### Principais vantagens do modelo:")
    st.markdown("""
    - Alta capacidade de generalização  
    - Bom desempenho em classificação multiclasse  
    - Robustez na captura de padrões não lineares  
    """)

    st.markdown("### Métrica principal")
    st.metric("Acurácia do modelo", "Alta ( > 85% )")

    st.markdown("""
    Os resultados indicam que variáveis como **peso**, **atividade física**,
    **hábitos alimentares** e **histórico familiar** são altamente relevantes
    na predição do nível de obesidade.
    """)
