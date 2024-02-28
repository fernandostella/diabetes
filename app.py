################################################
# Pós-Graduação em Big Data e Data Science
# Disciplina:   Implantação
# Professor:    Dr. Felipe de Morais
# Alunos:       Fernando Stella
#               Sidnei  
################################################
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import util
import data_handler
import pickle


#if not util.check_password():
#    st.stop()

# primeiro de tudo, carrega os dados dos testes para um dataframe
dados = data_handler.load_data()

# carrega o modelo de predição já treinado e validado
model = open('./model/model.pkl', 'rb')
model = pickle.load(model) 

# começa a estrutura da interface do sistema
st.title('Predição de Diagnóstico de Diabetes')

data_analyses_on = st.toggle('Exibir análise dos dados')

if data_analyses_on:
    # essa parte é só um exmplo de que é possível realizar diversas visualizações e plotagens com o streamlit
    st.header('Dados das análises de Diabetes - Dataframe')
    
    # exibe todo o dataframe dos dados de analises de diabetes
    st.dataframe(dados)

    # plota um histograma das idades dos pacientes
    st.header('Pacientes por Idade')
    fig = plt.figure()
    plt.hist(dados['Age'], bins=10)
    plt.xlabel('Idade')
    plt.ylabel('Quantidade')
    st.pyplot(fig)

    # plota um gráfico de barras com a contagem dos pacientes
    st.header('Resultados Positivos x Negativos')
    st.bar_chart(dados.Outcome.value_counts())
    
# Montagem da entrada de dados
st.header('Diagnóstico de Diabetes')

# Inputs:
# Pregnancies - int 
# Glucose - int
# BloodPressure - int
# SkinThickness - int
# Insulin - int
# BMI - float
# DiabetesPedigreeFunction - float
# Age - int


# Linha 1 de entrada de dados
col1, col2 = st.columns(2)
# Entrada de dados da quantidade de gestações da Paciente
with col1:    
    gestacoes = st.number_input('Nº Gestações', step=1, min_value=0)
# Nível de glicose do paciente
with col2:
    glicose = st.number_input('Glicose', step=10, min_value=0)

# Linha 2 de entrada de dados
col1, col2 = st.columns(2)
# Pressão Arterial
with col1:
    pressao = st.number_input('Pressão Arterial', step=10, min_value=0)    
# Espessura da pele 
with col2:
    pele = st.number_input('Espessura da pele', step=5, min_value=0)      

# Linha 3 de entrada de dados
col1, col2 = st.columns(2)
# Nível de insulina
with col1:
    insulina = st.number_input('Insulina', step=100, min_value=0)    
# IMC
with col2:
    imc = st.number_input('IMC',step=0.10, min_value=0.00)        

# Linha 4 de entrada de dados
col1, col2 = st.columns(2)
# Pontuação Histórico Familiar
with col1:
    hist = st.number_input('Pontuação Histórico Familiar',min_value=0.000, step=0.1)    
# Idade
with col2:
    idade = st.number_input('Idade', step=1, min_value=0)        

# Linha 5 - Botão de conformação
col1, col2 = st.columns(2)
with col1:
    submit = st.button('Analisar')

# Guarda os dados do paciente
paciente = {}
    
# verifica se o botão submit foi pressionado e se o campo Outcome está em cache
if submit or 'Outcome' in st.session_state:
    
    paciente = {
        'Pregnancies': gestacoes,
        'Glucose': glicose,
        'BloodPressure': pressao,
        'SkinThickness': pele,
        'Insulin': insulina,
        'BMI': imc,
        'DiabetesPedigreeFunction': hist,
        'Age': idade 
    }
    print(paciente)    

    # converte o registo do paciente para pandas dataframe
    values = pd.DataFrame([paciente])
    print(values) 

    # Analisa os dados para paciente para diagnósticar a diabetes
    results = model.predict(values)
    print(results)
    
    # Resultados:
    # 0) Não diagnosticado 
    # 1) Diagnosticado 
    
    if len(results) == 1:        
        diagnostico = int(results[0])        
        # Se o paciente é diabético
        if diagnostico == 1:            
            st.subheader('Resultado positivo para diabetes! 🤐🧁')
            if 'Outcome' not in st.session_state:
                 st.snow()
        else:            
            st.subheader('Resultado negatívo para diabetes! 🙌')
            if 'Outcome' not in st.session_state:
                st.balloons()
        
        # salva em cache da aplicação o resultado da predição do resultado do paciente
        st.session_state['Outcome'] = diagnostico
    
    # verifica se existe um passageiro e se já foi verificado se ele sobreviveu ou não
    if paciente and 'Outcome' in st.session_state:
        # se sim, pergunta ao usuário se a predição está certa e salva essa informação
        st.write("A predição está correta?")
        col1, col2, col3 = st.columns([1,1,5])
        with col1:
            correct_prediction = st.button('👍🏻')
        with col2:
            wrong_prediction = st.button('👎🏻')
        
        # exibe uma mensagem para o usuário agradecendo o feedback
        if correct_prediction or wrong_prediction:
            message = "Muito obrigado pelo feedback"
            if wrong_prediction:
                message += ", iremos usar esses dados para melhorar as predições"
            message += "."
            
            # adiciona no dict do paciente se o resultado está ou não correto
            if correct_prediction:
                paciente['CorrectPrediction'] = True
            elif wrong_prediction:
                paciente['CorrectPrediction'] = False
                
            # adiciona no dict o resultado do paciente
            paciente['Outcome'] = st.session_state['Outcome']
            
            # escreve a mensagem na tela
            st.write(message)
            print(message)
            
            # salva a predição no JSON para cálculo das métricas de avaliação do sistema
            #data_handler.save_prediction(paciente)
            
    st.write('')
    # adiciona um botão para permitir o usuário realizar uma nova análise
    col1, col2, col3 = st.columns(3)
    with col2:
        new_test = st.button('Iniciar Nova Análise')
        
        # se o usuário pressionar no botão e já existe um paciente, remove ele do cache
        if new_test and 'Outcome' in st.session_state:
            del st.session_state['Outcome']
            st.rerun()

# calcula e exibe as métricas de avaliação do modelo
# aqui, somente a acurária está sendo usada
# TODO: adicionar as mesmas métricas utilizadas na disciplina de treinamento e validação do modelo (recall, precision, F1-score)
accuracy_predictions_on = st.toggle('Exibir acurácia')

if accuracy_predictions_on:
    # pega todas as predições salvas no JSON
    predictions = data_handler.get_all_predictions()
    # salva o número total de predições realizadas
    num_total_predictions = len(predictions)
    
    # calcula o número de predições corretas e salva os resultados conforme as predições foram sendo realizadas
    accuracy_hist = [0]
    # salva o numero de predições corretas
    correct_predictions = 0
    # percorre cada uma das predições, salvando o total móvel e o número de predições corretas
    for index, paciente in enumerate(predictions):
        total = index + 1
        if paciente['CorrectPrediction'] == True:
            correct_predictions += 1
            
        # calcula a acurracia movel
        temp_accuracy = correct_predictions / total if total else 0
        # salva o valor na lista de historico de acuracias
        accuracy_hist.append(round(temp_accuracy, 2)) 
    
    # calcula a acuracia atual
    accuracy = correct_predictions / num_total_predictions if num_total_predictions else 0
    
    # exibe a acuracia atual para o usuário
    st.metric(label='Acurácia', value=round(accuracy, 2))
    # TODO: usar o attr delta do st.metric para exibir a diferença na variação da acurácia
    
    # exibe o histórico da acurácia
    st.subheader("Histórico de acurácia")
    st.line_chart(accuracy_hist)