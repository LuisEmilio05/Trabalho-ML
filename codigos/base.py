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

if __name__ == "__main__":

    df = pd.read_csv('spotify.csv', encoding='latin-1')

    # limpeza
    for c in ['in_deezer_playlists', 'in_shazam_charts']:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce')

    df['streams'] = pd.to_numeric(df['streams'], errors='coerce')
    df = df.dropna(subset=['streams']).reset_index(drop=True)

    # separar campos de entrada
    colunas_features = ['artist_count', 'released_year', 'released_month', 'released_day', 'in_spotify_playlists', 'in_spotify_charts', 'in_apple_playlists', 'in_apple_charts', 'in_deezer_playlists', 'in_deezer_charts', 'in_shazam_charts', 'bpm', 'danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']
    one_hot = pd.get_dummies(df[['key', 'mode']], dtype=float, dummy_na=True)
    X_temp = pd.concat ([df[colunas_features], one_hot], axis=1)
    X = X_temp.values

    # separar campo alvo
    Y = df['streams'].values.astype(float)

    # divisao dos dados (60% - treino, 20% - validação e 20% - teste)
    X_treino_val, X_teste, Y_treino_val, Y_teste = train_test_split(X, Y, test_size=0.20, random_state=42)

    X_treino, X_val, Y_treino, Y_val = train_test_split(X_treino_val, Y_treino_val, test_size=0.25, random_state=42)


    # Ridge & Lasso
    #alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    ridge = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RidgeCV(cv=5))
    ridge.fit(X_treino_val, Y_treino_val)

    lasso = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LassoCV(cv=5, random_state=42))
    lasso.fit(X_treino_val, Y_treino_val)

    for nome, modelo in [('Ridge', ridge), ('Lasso', lasso)]:
            y_pred = modelo.predict(X_teste)
            rmse = np.sqrt(mean_squared_error(Y_teste, y_pred))
            mae = mean_absolute_error(Y_teste, y_pred)
            r2 = r2_score(Y_teste, y_pred)

            print(f"-{nome}")
            print(f"Melhor alpha: {modelo[-1].alpha_}")
            print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")

            print("Colunas:")
            for feat, coef in zip(X_temp.columns, modelo[-1].coef_):
                print(f"  {feat}: {coef:.4f}")
            print()

    #MLP
    # Pre-Processamento de dados
    preproce = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())

    X_treino_proc = preproce.fit_transform(X_treino)
    X_val_proc = preproce.transform(X_val)
    X_teste_proc = preproce.transform(X_teste)

    y_m, y_s = Y_treino.mean(), Y_treino.std()

    X_treino_t = torch.tensor(X_treino_proc, dtype=torch.float32)
    Y_treino_t = torch.tensor((Y_treino - y_m) / y_s, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val_proc, dtype=torch.float32)
    Y_val_t = torch.tensor((Y_val - y_m) / y_s, dtype=torch.float32).unsqueeze(1)
    X_teste_t = torch.tensor(X_teste_proc, dtype=torch.float32)

    dataset_treino = TensorDataset(X_treino_t, Y_treino_t)
    loader = DataLoader(dataset_treino, batch_size=32, shuffle=True)

    # Definição e treinamento da MLP para regressão
    n_features = X_treino_proc.shape[1]
    model = nn.Sequential(nn.Linear(n_features, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),nn.Linear(32, 1))

    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 200
    paciencia = 15
    historico_loss = []
    historico_val = []
    melhor_val = float('inf')
    melhor_estado = None
    sem_melhora = 0


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


        model.eval()
        with torch.no_grad():
            loss_val = loss_fn(model(X_val_t), Y_val_t).item()
        historico_val.append(loss_val)
    
        if loss_val < melhor_val - 1e-4:
            melhor_val = loss_val
            sem_melhora = 0
            melhor_estado = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            sem_melhora += 1
            if sem_melhora >= paciencia:
                print(f"Early stopping na época {epoch + 1}")
                break
 
    model.load_state_dict(melhor_estado)   # volta aos pesos da melhor época

    # Curvas de perda (treino x validação)
    plt.plot(historico_loss, label='Treino')
    plt.plot(historico_val, label='Validação')
    plt.xlabel('Época')
    plt.ylabel('MSE (alvo padronizado)')
    plt.legend()
    plt.savefig('curva_loss.png', bbox_inches='tight')
    plt.close()


    # Avaliação da MLP no conjunto de teste
    model.eval()
    with torch.no_grad():
        y_pred_pad = model(X_teste_t).squeeze(1).numpy()
    y_pred_mlp = y_pred_pad * y_s + y_m

    rmse_mlp = np.sqrt(mean_squared_error(Y_teste, y_pred_mlp))
    mae_mlp = mean_absolute_error(Y_teste, y_pred_mlp)
    r2_mlp = r2_score(Y_teste, y_pred_mlp)

    q = np.quantile(Y_treino, [1/3, 2/3])
    bins = [-np.inf, q[0], q[1], np.inf]
    nomes_classes = ['Baixo', 'Médio', 'Alto']

    y_teste_classe = pd.cut(Y_teste, bins=bins, labels=nomes_classes)
    y_pred_classe = pd.cut(y_pred_mlp, bins=bins, labels=nomes_classes)

    cm = confusion_matrix(y_teste_classe, y_pred_classe, labels=nomes_classes)
    df_cm = pd.DataFrame(
        cm,
        index=[f"Real: {c}" for c in nomes_classes],
        columns=[f"Prev: {c}" for c in nomes_classes]
    )

    # prints
    print("- MLP (PyTorch)")
    print(f"RMSE: {rmse_mlp:.4f} | MAE: {mae_mlp:.4f} | R2: {r2_mlp:.4f}")
    print(df_cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nomes_classes)
    disp.plot(cmap='Blues')
    plt.savefig('matriz_confusao.png', bbox_inches='tight')
    plt.close()

