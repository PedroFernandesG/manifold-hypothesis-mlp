import json, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from experimento_mlp import make_data, MLP

AZUL, LARANJA, CINZA, TINTA = "#065A82", "#F2A65A", "#8A9BA6", "#142433"
from matplotlib.ticker import FuncFormatter
virg = FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ","))
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
os.makedirs("resultados", exist_ok=True)
b = json.load(open("resultados/base.json"))
g = json.load(open("resultados/grid.json"))
nomes = ["Entrada", "Oculta 1", "Oculta 2", "Oculta 3"]

# ---------------- Figura 1: sondas por camada ----------------
def painel(ax, rs, titulo):
    x = np.arange(4)
    for key, cor, ls, lab in [("lin", AZUL, "-", "Sonda linear (treinada)"),
                              ("ctrl_lin", AZUL, "--", "Sonda linear (pesos aleatórios)"),
                              ("knn", LARANJA, "-", "Sonda k-NN (treinada)"),
                              ("ctrl_knn", LARANJA, "--", "Sonda k-NN (pesos aleatórios)")]:
        a = np.array([r[key] for r in rs])
        ax.errorbar(x + (0.03 if "knn" in key else -0.03), a.mean(0), yerr=a.std(0), color=cor, ls=ls,
                    marker="o", ms=4, lw=1.6, capsize=2, label=lab, alpha=0.95)
    ax.set_xticks(x); ax.set_xticklabels(nomes)
    ax.set_ylim(0.4, 1.06); ax.set_ylabel("Acurácia de teste"); ax.yaxis.set_major_formatter(virg)
    ax.axhline(0.5, color=CINZA, lw=0.8, ls=":")
    ax.set_title(titulo, fontsize=9)

nt = np.array([r["net_test"] for r in b])
conv = [r for r, c in zip(b, nt >= 0.95) if c]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
painel(axs[0], b, f"(a) Todas as sementes (n = {len(b)})")
painel(axs[1], conv, f"(b) Apenas sementes convergidas (n = {len(conv)})")
axs[1].set_ylabel("")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.14, 1, 1))
fig.savefig("resultados/fig1_sondas.png", dpi=220); plt.close(fig)

# ---------------- Figura 2: geometria camada a camada (semente 0) ----------------
# Eixo x = direção da sonda linear (a que separa as classes); eixo y = 1º componente principal
# do restante (ortogonal a essa direção). Assim a figura mostra o que a sonda "enxerga".
from sklearn.linear_model import LogisticRegression
seed = 0
X, y = make_data("spirals", 1200, seed)
Xtr, ytr, Xte, yte = X[:800], y[:800], X[800:], y[800:]
net = MLP([2, 8, 8, 8, 1], "tanh", seed)
net.fit(Xtr, ytr, epochs=300, seed=seed)
A_tr, A_te = net.forward(Xtr)[1][:-1], net.forward(Xte)[1][:-1]
r0 = [r for r in b if r["seed"] == seed][0]
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.0))
for i, (ax, atr, ate) in enumerate(zip(axs, A_tr, A_te)):
    sonda = LogisticRegression(max_iter=2000).fit(atr, ytr)
    norma = np.linalg.norm(sonda.coef_[0])
    w = sonda.coef_[0] / norma                      # direção da sonda
    px = ate @ w
    py = PCA(1).fit_transform(ate - np.outer(px, w))[:, 0]
    thr = -sonda.intercept_[0] / norma              # posição da fronteira de decisão
    for c, cor in [(0, AZUL), (1, LARANJA)]:
        ax.scatter(px[yte == c], py[yte == c], s=5, color=cor, alpha=0.8, lw=0)
    ax.axvline(thr, color=TINTA, lw=1.0)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_color(CINZA); sp.set_linewidth(0.6)
    ax.set_title(f"{nomes[i]}\nsonda linear: " + f"{r0['lin'][i]:.2f}".replace(".", ","), fontsize=8)
fig.tight_layout()
fig.savefig("resultados/fig2_geometria.png", dpi=220); plt.close(fig)

# ---------------- Figura 3: largura x profundidade x ativação ----------------
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
larg, prof = [2, 4, 8, 16], [1, 2, 3, 4]
for ax, a, tit in zip(axs, ["tanh", "relu"], ["(a) tanh", "(b) ReLU"]):
    M = np.array([[np.mean([r["net_test"] for r in g if r["activation"] == a and r["width"] == w and r["depth"] == d])
                   for d in prof] for w in larg])
    im = ax.imshow(M, vmin=0.5, vmax=1.0, cmap="Blues", origin="lower", aspect="auto")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{M[i, j]:.2f}".replace(".", ","), ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 0.8 else TINTA)
    ax.set_xticks(range(4)); ax.set_xticklabels(prof); ax.set_yticks(range(4)); ax.set_yticklabels(larg)
    ax.set_xlabel("Camadas ocultas"); ax.set_title(tit, fontsize=9)
    for s in ax.spines.values(): s.set_visible(False)
axs[0].set_ylabel("Neurônios por camada")
cb = fig.colorbar(im, ax=axs, shrink=0.85, label="Acurácia de teste da rede"); cb.ax.yaxis.set_major_formatter(virg)
fig.savefig("resultados/fig3_grade.png", dpi=220, bbox_inches="tight"); plt.close(fig)
print("figuras ok")
