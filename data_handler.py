import pandas as pd
import numpy as np
import json

def load_data():
    dados = pd.read_csv('./data/diabetes.csv', sep = ';')
    return dados 

# retorna os dados validados pelo usuário
def get_all_predictions():
    data = None
    with open('./predictions.json', 'r') as f:
        data = json.load(f)        
    return data

# salva em arquivo JSON, testando antes se já não existe
def save_prediction(paciente):
    # faz a leitura dos dados já gravados
    data = get_all_predictions()
    # inclui a nova predição
    data.append(paciente)
    # Salva o arquivo        
    with open('predictions.json', 'w') as f:
        json.dump(data, f)
        