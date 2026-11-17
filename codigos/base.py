import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay

from func import standardization

if __name__ == "__main__":
    df = pd.read_csv('earthquake.csv')

    # separar campos de entrada
    colunas_features = ['tsunami', 'gap', 'depth', 'latitude', 'longitude']
    df = df.dropna(subset=['magnitude'])

    X = df[colunas_features].values

    # separar campo alvo
    Y = df['magnitude'].values

    # divisao dos dados (80% - treino, 20% - teste)
    X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.20, random_state=42)

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

    #MLP
    # Pre-Processamento de dados
    preproce = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())

    X_treino_proc = preproce.fit_transform(X_treino)
    X_teste_proc = preproce.transform(X_teste)

    X_treino_t = torch.tensor(X_treino_proc, dtype=torch.float32)
    Y_treino_t = torch.tensor(Y_treino, dtype=torch.float32).unsqueeze(1)
    X_teste_t = torch.tensor(X_teste_proc, dtype=torch.float32)

    dataset_treino = TensorDataset(X_treino_t, Y_treino_t)
    loader = DataLoader(dataset_treino, batch_size=32, shuffle=True)

    # Definição e treinamento da MLP para regressão
    n_features = X_treino_proc.shape[1]
    model = nn.Sequential(nn.Linear(n_features, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),nn.Linear(32, 1))

    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    historico_loss = []

    for epoch in range(epochs):
        model.train()
        loss_epoca = 0.0

        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_epoca += loss.item() * xb.size(0)

        historico_loss.append(loss_epoca / len(dataset_treino))

    # Avaliação da MLP no conjunto de teste
    model.eval()
    with torch.no_grad():
        y_pred_mlp = model(X_teste_t).squeeze(1).numpy()

    bins = [-np.inf, 6.0, 7.0, np.inf]
    nomes_classes = ['Moderado (<6.0)', 'Forte (6.0-6.9)', 'Maior (>=7.0)']

    y_teste_classe = pd.cut(Y_teste, bins=bins, labels=nomes_classes)
    y_pred_classe = pd.cut(y_pred_mlp, bins=bins, labels=nomes_classes)

    cm = confusion_matrix(y_teste_classe, y_pred_classe, labels=nomes_classes)
    df_cm = pd.DataFrame(
        cm,
        index=[f"Real: {c}" for c in nomes_classes],
        columns=[f"Prev: {c}" for c in nomes_classes]
    )

    rmse_mlp = np.sqrt(mean_squared_error(Y_teste, y_pred_mlp))
    mae_mlp = mean_absolute_error(Y_teste, y_pred_mlp)
    r2_mlp = r2_score(Y_teste, y_pred_mlp)

    # prints
    print("- MLP (PyTorch)")
    print(f"RMSE: {rmse_mlp:.4f} | MAE: {mae_mlp:.4f} | R2: {r2_mlp:.4f}")
    print(df_cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nomes_classes)
    disp.plot(cmap='Blues')
    plt.savefig('matriz_confusao.png', bbox_inches='tight')
    plt.close()
