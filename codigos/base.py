import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix

from func import standardization

if __name__ == "__main__":
    df = pd.read_csv('earthquake.csv')

    # separar campos de entrada

    # cdi, mmi, alert, sig -- consequencias do terremoto
    # magType, location, continent, country, nst, dmin ??
    colunas_features = ['tsunami', 'gap', 'depth', 'latitude', 'longitude']
    df = df.dropna(subset=['magnitude'])
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

    # Ridge & Lasso
    alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    ridge = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RidgeCV(alphas=alphas, cv=5))
    ridge.fit(X_treino, Y_treino)

    lasso = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LassoCV(alphas=alphas, cv=5, random_state=42))
    lasso.fit(X_treino, Y_treino)

    for nome, modelo in [('Ridge', ridge), ('Lasso', lasso)]:
            y_pred = modelo.predict(X_teste)
            rmse = np.sqrt(mean_squared_error(Y_teste, y_pred))
            mae = mean_absolute_error(Y_teste, y_pred)
            r2 = r2_score(Y_teste, y_pred)

            print(f"-{nome}")
            print(f"Melhor alpha: {modelo[-1].alpha_}")
            print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")

            print("Colunas:")
            for feat, coef in zip(colunas_features, modelo[-1].coef_):
                print(f"  {feat}: {coef:.4f}")
            print()
