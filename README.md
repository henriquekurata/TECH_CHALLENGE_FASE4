# 🧠 Sistema Preditivo de Obesidade

Este projeto foi desenvolvido como parte do **Tech Challenge – Fase 4 (Data Analytics)**,
com o objetivo de criar um sistema preditivo para **auxiliar a equipe médica
na identificação do nível de obesidade de pacientes**, utilizando Machine Learning.

---

## 🎯 Objetivo
Desenvolver um modelo de Machine Learning capaz de prever o nível de obesidade
com base em dados físicos, demográficos e comportamentais, entregando uma
aplicação interativa para apoio à tomada de decisão clínica.

---

## 📊 Base de Dados
Dataset: `Obesity.csv`

Variáveis consideradas:
- Dados físicos: idade, altura, peso
- Hábitos alimentares
- Frequência de atividade física
- Estilo de vida e transporte
- Histórico familiar

Variável alvo:
- `Obesity_level` (classificação multiclasse)

---

## ⚙️ Pipeline de Machine Learning
O projeto utiliza um pipeline completo com:
- Tratamento de variáveis numéricas (StandardScaler)
- Codificação de variáveis categóricas (OneHotEncoder)
- Modelagem integrada ao pré-processamento

Modelos testados:
- Regressão Logística
- Random Forest
- Gradient Boosting

Após comparação, o **Gradient Boosting Classifier** foi selecionado
por apresentar o melhor desempenho, com acurácia superior a 85%.

---

## 🧠 Sistema Preditivo
A aplicação permite que o usuário insira dados de um paciente
e receba, em tempo real, a previsão do nível de obesidade.

⚠️ Este sistema é uma ferramenta de **apoio à decisão clínica**
e não substitui o diagnóstico médico.

---

## 📈 Dashboard Analítico
O sistema conta com uma visão analítica interativa que apresenta:
- Distribuição dos níveis de obesidade
- Relação entre atividade física e obesidade
- Relação entre idade, peso e nível de obesidade
- Principais insights do modelo

---

## 🚀 Deploy
A aplicação foi desenvolvida em **Streamlit** e está disponível em:

🔗 **Link da aplicação:**  
(cole aqui o link do Streamlit Cloud)

---

## 🎥 Apresentação
Foi gravado um vídeo demonstrando:
- Estratégia adotada
- Análise dos dados
- Funcionamento do sistema preditivo
- Impacto do projeto para a área da saúde

---

## 🛠️ Tecnologias Utilizadas
- Python
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
