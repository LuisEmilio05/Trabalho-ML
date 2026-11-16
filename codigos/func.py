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

def fit_standardization (x):
    m = mean(x)
    d = std_deviation(x, m)
    return m, d

def apply_standardization (x, m, d):
    return [(i - m) / d for i in x]

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

