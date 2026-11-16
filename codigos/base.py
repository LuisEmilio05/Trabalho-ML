import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from func import standardization

if __name__ == "__main__":
    df = pd.read_csv('earthquake.csv')

    # separar campos de entrada

    # cdi, mmi, alert, sig -- consequencias do terremoto
    # magType, location, continent, country, nst, dmin ??
    colunas_features = ['tsunami', 'gap', 'depth', 'latitude', 'longitude']
    #codificacao_alert = pd.get_dummies(df['alert'], dtype=float)
    #X_temp = pd.concat ([df[colunas_features], codificacao_alert], axis=1)
    #X = X_temp.values
    X = df[colunas_features].values

    # separar campo alvo
    Y = df['magnitude'].values

    # divisao dos dados (80% - treino, 20% - teste)
    X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.20, random_state=42)
    

    # normalização
    X_treino = standardization(X_treino)

    # chama os modelos
