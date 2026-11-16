from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV
import torch, torch.nn as nn

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Pré processamento
def mean (x):
    sum = 0
    for i in x:
        sum += i
    mean = sum / len(x)
    return mean

def std_deviation (x, mean):
    sum = 0
    for i in x:
        sum += (i - mean)**2
    fraction = sum / len(x)
    deviation = fraction**(1/2)
    return deviation

def standardization (x) :
    m = mean(x)
    dev = std_deviation(x, m)
    z = []
    for i in x:
        z.append((i - m) / dev)  
    return z

def centred (x, mean):
    centred_x = []
    for i in x:
        centred_x.append((i - mean))

    return centred_x

def norm(x):
    norm = 0
    for i in x:
        norm += (i)**2
    norm = norm**(1/2)

    return norm

def dot_product(x, y):
    dot = 0
    for i in range(len(x)):
        dot += (x[i] * y[i])

    return dot

def correlation(x, y):
    xc = centred(x, mean(x))
    yc = centred(y, mean(y))
    return dot_product(xc, yc) / (norm(xc) * norm(yc))


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

