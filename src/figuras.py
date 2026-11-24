"""Gera as figuras do artigo a partir de resultados/*.json"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from experimento_mlp import split, MLP

AZUL, LARANJA, CINZA, TINTA = "#065A82", "#F2A65A", "#8A9BA6", "#142433"
virg = FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ","))
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
os.makedirs("resultados", exist_ok=True)
L = lambda n: json.load(open(f"resultados/{n}.json"))
base = L("base")
nomes = ["Entrada", "Oculta 1", "Oculta 2", "Oculta 3"]

# ---------------- Figura 1: sondas por camada (tanh, lr 0,01) ----------------
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

tanh = [r for r in base if r["activation"] == "tanh"]
conv = [r for r in tanh if r["converged"]]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
painel(axs[0], tanh, f"(a) Todas as sementes (n = {len(tanh)})")
painel(axs[1], conv, f"(b) Apenas sementes convergidas (n = {len(conv)})")
axs[1].set_ylabel("")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.14, 1, 1))
fig.savefig("resultados/fig1_sondas.png", dpi=220); plt.close(fig)

# ---------------- Figura 2: geometria camada a camada (1ª semente convergida) ----------------
seed = conv[0]["seed"]
Xtr, ytr, Xva, yva, Xte, yte = split("spirals", seed, 0.05)
net = MLP([2, 8, 8, 8, 1], "tanh", seed)
net.fit(Xtr, ytr, epochs=300, lr=0.01, seed=seed)
A_tr, A_te = net.forward(Xtr)[1][:-1], net.forward(Xte)[1][:-1]
r0 = conv[0]
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.0))
for i, (ax, atr, ate) in enumerate(zip(axs, A_tr, A_te)):
    sonda = LogisticRegression(max_iter=3000).fit(atr, ytr)
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
print("semente da Figura 2:", seed)

# ---------------- Figura 3: taxa de aprendizado ----------------
lr_rows = L("lr_tanh") + L("lr_relu")
lrs = [0.001, 0.003, 0.01, 0.03, 0.1]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7))
for a, cor in [("tanh", AZUL), ("relu", LARANJA)]:
    conv_rate = [100 * np.mean([r["converged"] for r in lr_rows if r["activation"] == a and r["lr"] == lr]) for lr in lrs]
    acc = [np.mean([r["net_test"] for r in lr_rows if r["activation"] == a and r["lr"] == lr]) for lr in lrs]
    axs[0].plot(lrs, conv_rate, marker="o", color=cor, label="ReLU" if a == "relu" else "tanh")
    axs[1].plot(lrs, acc, marker="o", color=cor, label="ReLU" if a == "relu" else "tanh")
for ax in axs:
    ax.set_xscale("log"); ax.set_xlabel("Taxa de aprendizado")
    ax.set_xticks(lrs); ax.set_xticklabels([str(x).replace(".", ",") for x in lrs])
    ax.minorticks_off()
axs[0].set_ylabel("Sementes convergidas (%)"); axs[0].set_title("(a) Convergência", fontsize=9)
axs[1].set_ylabel("Acurácia de teste média"); axs[1].set_title("(b) Desempenho", fontsize=9)
axs[1].set_yticks([0.7, 0.8, 0.9, 1.0]); axs[1].yaxis.set_major_formatter(virg); axs[1].legend(frameon=False)
fig.tight_layout()
fig.savefig("resultados/fig3_lr.png", dpi=220); plt.close(fig)

# ---------------- Figura 4: grade com taxa escolhida pela validação ----------------
g = []
for f in sorted(glob.glob("resultados/grid_*.json")): g += json.load(open(f))
larg, prof = [2, 4, 8, 16], [1, 2, 3, 4]
def melhor(a, w, d):
    cand = {}
    for lr in [0.003, 0.01, 0.03]:
        rs = [r for r in g if r["activation"] == a and r["width"] == w and r["depth"] == d and r["lr"] == lr]
        cand[lr] = (np.mean([r["net_val"] for r in rs]), rs)
    lrb = max(cand, key=lambda k: cand[k][0])
    return np.mean([r["net_test"] for r in cand[lrb][1]])
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
for ax, a, tit in zip(axs, ["tanh", "relu"], ["(a) tanh", "(b) ReLU"]):
    M = np.array([[melhor(a, w, d) for d in prof] for w in larg])
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
fig.savefig("resultados/fig4_grade.png", dpi=220, bbox_inches="tight"); plt.close(fig)

# ---------------- Figura 5: tanh x ReLU por camada, espirais e luas ----------------
dados = L("data_tanh") + L("data_relu")
def sel(ds, nz, a):
    pool = base if (ds, nz) == ("spirals", 0.05) else dados
    return [r for r in pool if r["dataset"] == ds and r["noise"] == nz and r["activation"] == a and r["converged"]]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
for ax, (ds, nz, tit) in zip(axs, [("spirals", 0.05, "(a) Espirais (ruído 0,05)"), ("moons", 0.2, "(b) Luas (ruído 0,2)")]):
    ns = {}
    for a, cor, nome in [("tanh", AZUL, "tanh"), ("relu", LARANJA, "ReLU")]:
        rs = sel(ds, nz, a); ns[nome] = len(rs)
        for key, ls, lab in [("lin", "-", "sonda linear"), ("knn", ":", "sonda k-NN")]:
            m = np.array([r[key] for r in rs])
            ax.errorbar(np.arange(4), m.mean(0), yerr=m.std(0), color=cor, ls=ls, marker="o", ms=4, lw=1.6,
                        capsize=2, label=f"{nome}, {lab}")
    ax.set_xticks(range(4)); ax.set_xticklabels(nomes)
    ax.set_title(f"{tit}\nsementes convergidas: tanh {ns['tanh']}, ReLU {ns['ReLU']}", fontsize=8.5)
    ax.set_ylim(0.5, 1.06); ax.yaxis.set_major_formatter(virg)
axs[0].set_ylabel("Acurácia de teste")
h2, l2 = axs[1].get_legend_handles_labels()
fig.legend(h2, l2, loc="lower center", ncol=2, frameon=False, fontsize=7.5)
fig.tight_layout(rect=(0, 0.17, 1, 1))
fig.savefig("resultados/fig5_ativacao_dados.png", dpi=220); plt.close(fig)
print("figuras ok")
