import json, os, sys, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_circles

def make_spirals(n, rng, turns=1.5, noise=0.05):
    y = rng.integers(0, 2, n)
    u = rng.uniform(0.12, 1.0, n)
    theta = u * turns * 2 * np.pi + y * np.pi
    X = np.stack([u * np.cos(theta), u * np.sin(theta)], axis=1)
    X += rng.normal(0, noise, X.shape)
    return X, y.astype(float)

def make_data(name, n, seed):
    rng = np.random.default_rng(seed)
    if name == "spirals":
        return make_spirals(n, rng)
    if name == "circles": 
        X, y = make_circles(n_samples=n, noise=0.05, factor=0.4, random_state=seed)
        return X, y.astype(float)
    raise ValueError(name)

def act(name):
    if name == "tanh":
        return np.tanh, lambda z: 1 - np.tanh(z) ** 2
    if name == "relu":
        return lambda z: np.maximum(0, z), lambda z: (z > 0).astype(float)
    raise ValueError(name)

class MLP:
    def __init__(self, sizes, activation, seed):
        """sizes = [n_in, h1, ..., hk, 1]"""
        rng = np.random.default_rng(seed)
        self.f, self.df = act(activation)
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            lim = np.sqrt(6 / (sizes[i] + sizes[i + 1]))        # Glorot
            self.W.append(rng.uniform(-lim, lim, (sizes[i], sizes[i + 1])))
            self.b.append(np.zeros(sizes[i + 1]))

    def forward(self, X):
        a, zs, acts = X, [], [X]
        for l in range(len(self.W)):
            z = a @ self.W[l] + self.b[l]
            zs.append(z)
            a = 1 / (1 + np.exp(-z)) if l == len(self.W) - 1 else self.f(z)
            acts.append(a)
        return zs, acts

    def fit(self, X, y, epochs=300, lr=0.01, batch=64, seed=0):
        rng = np.random.default_rng(seed)
        y = y.reshape(-1, 1)
        params = self.W + self.b
        m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
        t = 0
        for _ in range(epochs):
            idx = rng.permutation(len(X))
            for s in range(0, len(X), batch):
                bi = idx[s:s + batch]
                zs, acts = self.forward(X[bi])
                delta = acts[-1] - y[bi]                           # delta(L) = y_hat - y
                gW, gb = [None] * len(self.W), [None] * len(self.W)
                for l in range(len(self.W) - 1, -1, -1):
                    gW[l] = acts[l].T @ delta / len(bi)
                    gb[l] = delta.mean(axis=0)
                    if l > 0:
                        delta = (delta @ self.W[l].T) * self.df(zs[l - 1])
                grads = gW + gb
                t += 1
                for i, (p, g) in enumerate(zip(params, grads)):    # Adam
                    m[i] = 0.9 * m[i] + 0.1 * g
                    v[i] = 0.999 * v[i] + 0.001 * g * g
                    p -= lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)

    def accuracy(self, X, y):
        return float(((self.forward(X)[1][-1].ravel() >= 0.5) == y).mean())

# ---------------------------------------------------------------- sondas
def probes(net, Xtr, ytr, Xte, yte, k=5):
    """Acurácia das sondas linear e k-NN em cada camada (0 = entrada ... última oculta)."""
    A_tr, A_te = net.forward(Xtr)[1][:-1], net.forward(Xte)[1][:-1]
    lin, knn = [], []
    for atr, ate in zip(A_tr, A_te):
        if np.allclose(atr.std(axis=0), 0):                      # camada "morta" (ReLU)
            lin.append(0.5); knn.append(0.5); continue
        lin.append(LogisticRegression(max_iter=2000).fit(atr, ytr).score(ate, yte))
        knn.append(KNeighborsClassifier(n_neighbors=k).fit(atr, ytr).score(ate, yte))
    return lin, knn

# ---------------------------------------------------------------- uma execução
def run_one(dataset, width, depth, activation, seed, epochs=300, n_train=800, n_test=400):
    X, y = make_data(dataset, n_train + n_test, seed)
    Xtr, ytr, Xte, yte = X[:n_train], y[:n_train], X[n_train:], y[n_train:]
    net = MLP([2] + [width] * depth + [1], activation, seed)
    ctrl_lin, ctrl_knn = probes(net, Xtr, ytr, Xte, yte)         # controle: pesos aleatórios
    net.fit(Xtr, ytr, epochs=epochs, seed=seed)
    lin, knn = probes(net, Xtr, ytr, Xte, yte)
    return dict(dataset=dataset, width=width, depth=depth, activation=activation, seed=seed,
                net_train=net.accuracy(Xtr, ytr), net_test=net.accuracy(Xte, yte),
                lin=lin, knn=knn, ctrl_lin=ctrl_lin, ctrl_knn=ctrl_knn)

if __name__ == "__main__":
    which = sys.argv[1]            
    os.makedirs("resultados", exist_ok=True)
    out = []
    t0 = time.time()
    if which == "base":         
        for s in range(10):
            out.append(run_one("spirals", 8, 3, "tanh", s))
    elif which == "grid":     
        for act_ in ["tanh", "relu"]:
            for w in [2, 4, 8, 16]:
                for d in [1, 2, 3, 4]:
                    for s in range(5):
                        out.append(run_one("spirals", w, d, act_, s))
    elif which == "circles":   
        for w in [2, 3, 4]:
            for s in range(5):
                out.append(run_one("circles", w, 3, "tanh", s))
    json.dump(out, open(f"resultados/{which}.json", "w"))
    print(which, len(out), "execuções em", round(time.time() - t0, 1), "s")
