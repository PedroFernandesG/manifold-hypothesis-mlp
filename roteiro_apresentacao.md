# Roteiro sugerido — 15 a 20 minutos

## 1. Título — 30 s
Apresente o trabalho, os autores e a instituição. Destaque que o objetivo é observar a geometria interna de MLPs em vez de tratá-las apenas como caixa-preta.

## 2. Por que analisar o interior da rede? — 1 min
Explique o contraste entre capacidade prática e falta de explicabilidade. Formule a Hipótese do Manifold e a pergunta central.

## 3. A rede desenrola uma representação — 1 min
Use a analogia de esticar, dobrar e projetar uma variedade. Deixe claro que é uma interpretação geométrica, não uma prova direta da hipótese.

## 4. Formalização — 1 min
Percorra as equações: pré-ativação, ativação, saída sigmoide e composição da rede. Compare tanh e ReLU.

## 5. Treinamento — 1 min
Explique entropia cruzada, Adam, lotes e retropropagação. Ressalte que os parâmetros são a maneira pela qual a transformação geométrica é aprendida.

## 6. Metodologia — 1,5 min
Mostre o protocolo completo: dados sintéticos, divisão 800/200/400, arquitetura principal, critérios de convergência e controles.

## 7. Sondas — 1,5 min
Explique que uma sonda linear usa regressão logística sobre ativações congeladas. k-NN mede separação local. Nenhuma sonda altera a rede original.

## 8. Resultado principal — 1,5 min
Destaque 23/30 convergências com tanh. Leia a curva entrada → 0,61; última oculta → 0,99. Mostre que o ganho não é uniforme.

## 9. Geometria — 1 min
Explique a projeção da Figura 2. A linha vertical é a fronteira da sonda linear; o eixo vertical apenas espalha os pontos.

## 10. Taxa de aprendizado — 1,5 min
Compare lr 0,001 e 0,03 na tanh. Mostre que falhas não são necessariamente capacidade insuficiente: otimização lenta pode explicar convergência.

## 11. Largura e profundidade — 1,5 min
Leia a grade. Diferencie insuficiência de largura e ganhos adicionais de profundidade. A largura 2 falha sistematicamente.

## 12. Outros dados — 1 min
Compare espirais, luas e ruído alto. Conecte ruído alto à sobreposição das classes e ao limite da informação disponível.

## 13. Controles — 1,5 min
Discuta círculos concêntricos e redes aleatórias. Mostre que 512 neurônios tanh podem separar por projeção aleatória, mas isso não elimina o papel do treinamento em redes estreitas.

## 14. Síntese — 1 min
Feche os cinco resultados mais fortes e distingua evidência experimental de interpretação geométrica.

## 15. Discussão e limitações — 1 min
Reconheça que os dados são sintéticos, bidimensionais e limitados a um número pequeno de sementes e hiperparâmetros.

## 16. Conclusão — 1 min
Retome a resposta à pergunta central: redes convergidas transformam representações para uma fronteira linear, mas os resultados dependem de otimização, arquitetura e dados. Indique trabalhos futuros.
