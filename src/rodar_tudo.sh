#!/bin/bash
# Reproduz todos os resultados e figuras do artigo (cerca de 20 minutos em 1 CPU).
set -e
python3 -W ignore experimento_mlp.py base                       # experimento principal (tanh e ReLU, lr 0,01)
python3 -W ignore experimento_mlp.py base tanh 0.03             # sensibilidade: tanh com lr 0,03
python3 -W ignore experimento_mlp.py long tanh                  # 1000 épocas
for a in tanh relu; do python3 -W ignore experimento_mlp.py lr $a; done   # varredura de taxa de aprendizado
for a in tanh relu; do python3 -W ignore experimento_mlp.py data $a; done  # luas e espirais mais ruidosas
python3 -W ignore experimento_mlp.py circles                    # classe circundada por outra
python3 -W ignore experimento_mlp.py randscale                  # controle com pesos aleatórios
for a in tanh relu; do for lr in 0.003 0.01 0.03; do
  python3 -W ignore experimento_mlp.py grid_lr $a $lr           # grade largura x profundidade
done; done
python3 -W ignore figuras.py
echo CONCLUIDO
