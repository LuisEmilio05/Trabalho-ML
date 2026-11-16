from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV
import torch, torch.nn as nn

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Treinamento

# Ridge and Lasso
alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas, cv=5))
ridge.fit(X_train, y_train)
lasso = make_pipeline(StandardScaler(), LassoCV(cv=5))
lasso.fit(X_train, y_train)
coef = lasso[-1].coef_
print((coef != 0).sum(), "features kept")

# MLP
model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters())
for xb, yb in loader: # mini-batches
    loss = loss_fn(model(xb), yb)
    opt.zero_grad()
    loss.backward() # backpropagation
    opt.step() # gradient step

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True,
random_state=0)
scores = cross_val_score(model, X, y, cv=cv)
print(k, scores.mean(), scores.std())

