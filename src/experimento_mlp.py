"""
Experimentos: separabilidade linear camada a camada em MLPs (versão final).

A MLP é implementada em numpy seguindo as equações do artigo:
  z(l) = W(l) a(l-1) + b(l);  a(l) = sigma(z(l));  y_hat = sigmoide(z(L))
  perda = entropia cruzada binária;  delta(L) = y_hat - y;
  delta(l) = (W(l+1)^T delta(l+1)) ⊙ sigma'(z(l))
(Convenção de código: exemplos em LINHAS, logo z = a @ W + b, equivalente à forma do artigo.)

Protocolo: 800 exemplos de treino, 200 de validação e 400 de teste, gerados a cada semente.
 - a rede é treinada no treino;
 - a validação é usada para (i) definir se a rede "convergiu" (acurácia de validação >= 0,95) e
   (ii) escolher a taxa de aprendizado na grade; o teste só é usado para reportar resultados;
 - as sondas (regressão logística e k-NN) são treinadas nas ativações do treino e avaliadas no teste.

Uso:  python experimento_mlp.py <base|long|lr|grid_lr|data|circles|randscale> [tanh|relu] [lr]
"""
import json, os, sys, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_circles, make_moons

N_TR, N_VA, N_TE = 800, 200, 400
LIMIAR_CONV = 0.95

# ---------------------------------------------------------------- dados
def make_spirals(n, rng, turns=1.5, noise=0.05):
    """Duas espirais entrelaçadas (classes 0 e 1)."""
    y = rng.integers(0, 2, n)
    u = rng.uniform(0.12, 1.0, n)
    theta = u * turns * 2 * np.pi + y * np.pi
    X = np.stack([u * np.cos(theta), u * np.sin(theta)], axis=1)
    X += rng.normal(0, noise, X.shape)
    return X, y.astype(float)

def make_data(name, n, seed, noise=0.05):
    rng = np.random.default_rng(seed)
    if name == "spirals":
        return make_spirals(n, rng, noise=noise)
    if name == "moons":
        X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
        return X, y.astype(float)
    if name == "circles":   # uma classe circundada por outra
        X, y = make_circles(n_samples=n, noise=noise, factor=0.4, random_state=seed)
        return X, y.astype(float)
    raise ValueError(name)

def split(dataset, seed, noise):
    X, y = make_data(dataset, N_TR + N_VA + N_TE, seed, noise)
    a, b = N_TR, N_TR + N_VA
    return X[:a], y[:a], X[a:b], y[a:b], X[b:], y[b:]

# ---------------------------------------------------------------- MLP
def act(name):
    if name == "tanh":
        return np.tanh, lambda z: 1 - np.tanh(z) ** 2
    if name == "relu":
        return lambda z: np.maximum(0, z), lambda z: (z > 0).astype(float)
    raise ValueError(name)

class MLP:
    def __init__(self, sizes, activation, seed, scale=1.0):
        """sizes = [n_in, h1, ..., hk, 1]; scale multiplica o intervalo de Glorot (padrão 1)"""
        rng = np.random.default_rng(seed)
        self.f, self.df = act(activation)
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            lim = scale * np.sqrt(6 / (sizes[i] + sizes[i + 1]))   # Glorot
            self.W.append(rng.uniform(-lim, lim, (sizes[i], sizes[i + 1])))
            self.b.append(np.zeros(sizes[i + 1]))

    def forward(self, X):
        a, zs, acts = X, [], [X]
        for l in range(len(self.W)):
            z = a @ self.W[l] + self.b[l]
            zs.append(z)
            a = 1 / (1 + np.exp(-np.clip(z, -500, 500))) if l == len(self.W) - 1 else self.f(z)
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
def probes(net, Xtr, ytr, Xte, yte, k=5, do_knn=True, standardize=False):
    """Acurácia das sondas linear e k-NN em cada camada (0 = entrada ... última oculta)."""
    A_tr, A_te = net.forward(Xtr)[1][:-1], net.forward(Xte)[1][:-1]
    lin, knn = [], []
    for atr, ate in zip(A_tr, A_te):
        if np.allclose(atr.std(axis=0), 0):                      # camada "morta" (ReLU)
            lin.append(0.5); knn.append(0.5); continue
        if standardize:                                           # features em escala muito diferente
            sc = StandardScaler().fit(atr); atr, ate = sc.transform(atr), sc.transform(ate)
        lin.append(LogisticRegression(max_iter=3000).fit(atr, ytr).score(ate, yte))
        knn.append(KNeighborsClassifier(n_neighbors=k).fit(atr, ytr).score(ate, yte) if do_knn else None)
    return lin, knn

# ---------------------------------------------------------------- uma execução
def run_one(dataset, width, depth, activation, seed, lr=0.01, epochs=300, noise=0.05, with_probes=True):
    Xtr, ytr, Xva, yva, Xte, yte = split(dataset, seed, noise)
    net = MLP([2] + [width] * depth + [1], activation, seed)
    out = dict(dataset=dataset, noise=noise, width=width, depth=depth, activation=activation,
               seed=seed, lr=lr, epochs=epochs)
    if with_probes:
        out["ctrl_lin"], out["ctrl_knn"] = probes(net, Xtr, ytr, Xte, yte)   # controle: pesos aleatórios
    net.fit(Xtr, ytr, epochs=epochs, lr=lr, seed=seed)
    out.update(net_train=net.accuracy(Xtr, ytr), net_val=net.accuracy(Xva, yva), net_test=net.accuracy(Xte, yte))
    out["converged"] = bool(out["net_val"] >= LIMIAR_CONV)
    if with_probes:
        out["lin"], out["knn"] = probes(net, Xtr, ytr, Xte, yte)
    return out

def run_random(dataset, width, depth, activation, seed, noise=0.05, scale=1.0):
    """Rede SEM treino (pesos aleatórios) com largura variável: sonda linear em cada camada."""
    Xtr, ytr, _, _, Xte, yte = split(dataset, seed, noise)
    net = MLP([2] + [width] * depth + [1], activation, seed, scale=scale)
    lin, _ = probes(net, Xtr, ytr, Xte, yte, do_knn=False, standardize=True)
    return dict(dataset=dataset, noise=noise, width=width, depth=depth, activation=activation, seed=seed, scale=scale, lin=lin)

def save(name, rows):
    os.makedirs("resultados", exist_ok=True)
    json.dump(rows, open(f"resultados/{name}.json", "w"))

if __name__ == "__main__":
    which = sys.argv[1]
    sel_act = sys.argv[2] if len(sys.argv) > 2 else None      # opcional: restringe a ativação
    sel_lr = float(sys.argv[3]) if len(sys.argv) > 3 else None  # opcional: restringe a taxa de aprendizado
    acts = [sel_act] if sel_act else ["tanh", "relu"]
    out, t0, tag = [], time.time(), which
    if which == "base":          # espirais, 3x8, lr 0,01 (fixada a priori), 30 sementes, com sondas
        for a in acts:
            for s in range(30):
                out.append(run_one("spirals", 8, 3, a, s, lr=sel_lr or 0.01))
    elif which == "long":        # checagem: 1000 épocas em vez de 300 (lr 0,01), 30 sementes
        for a in acts:
            for s in range(30):
                out.append(run_one("spirals", 8, 3, a, s, lr=0.01, epochs=1000, with_probes=False))
    elif which == "randscale":   # controle aleatório com escala de pesos maior que a de Glorot
        for a in acts:
            for w in [8, 512]:
                for sc in [1, 3, 10, 30]:
                    for s in range(30):
                        out.append(run_random("spirals", w, 3, a, s, scale=sc))
    elif which == "lr":          # varredura de taxa de aprendizado na configuração-base, 30 sementes
        for a in acts:
            for lr in [0.001, 0.003, 0.01, 0.03, 0.1]:
                for s in range(30):
                    out.append(run_one("spirals", 8, 3, a, s, lr=lr, with_probes=False))
    elif which == "grid_lr":     # grade largura x profundidade x taxa de aprendizado, 10 sementes
        lrs = [sel_lr] if sel_lr else [0.003, 0.01, 0.03]
        tag = f"grid_{'_'.join(acts)}" + (f"_{sel_lr}" if sel_lr else "")
        for a in acts:
            for lr in lrs:
                for w in [2, 4, 8, 16]:
                    for d in [1, 2, 3, 4]:
                        for s in range(10):
                            out.append(run_one("spirals", w, d, a, s, lr=lr, with_probes=False))
    elif which == "data":        # outros conjuntos / mais ruído, config-base, 30 sementes, com sondas
        for ds, nz in [("moons", 0.2), ("spirals", 0.15), ("spirals", 0.25)]:
            for a in acts:
                for s in range(30):
                    out.append(run_one(ds, 8, 3, a, s, lr=0.01, noise=nz))
    elif which == "circles":     # uma classe circundada por outra: larguras 2, 3 e 4
        for w in [2, 3, 4]:
            for s in range(10):
                out.append(run_one("circles", w, 3, "tanh", s, lr=0.01))
    suffix = (f"_{sel_act}" if sel_act else "") + (f"_{sel_lr}" if (sel_lr and which == "base") else "")
    if tag == which and suffix:
        tag = which + suffix                 # ex.: lr_tanh, data_relu, base_tanh_0.03
    save(tag, out)
    print(tag, len(out), "execuções em", round(time.time() - t0, 1), "s")
