from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from PIL import Image
import fitz
import os

OUT = "apresentacao_artigo.pptx"
NOTES = "roteiro_apresentacao.md"
PDF = "artigo_completo_pfg_dimn.pdf"

BLUE = RGBColor(6, 90, 130)
ORANGE = RGBColor(242, 166, 90)
INK = RGBColor(20, 36, 51)
MUTED = RGBColor(100, 116, 128)
LIGHT = RGBColor(239, 246, 249)
PALE = RGBColor(247, 249, 250)
WHITE = RGBColor(255, 255, 255)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)


def add_bg(slide, color=WHITE):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_header(slide, title, kicker=None):
    slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.08)).fill.solid()
    slide.shapes[-1].fill.fore_color.rgb = BLUE
    slide.shapes[-1].line.fill.background()
    if kicker:
        tb = slide.shapes.add_textbox(Inches(0.65), Inches(0.35), Inches(11.9), Inches(0.25))
        p = tb.text_frame.paragraphs[0]
        p.text = kicker.upper()
        p.font.name = "Aptos"
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = ORANGE
    tb = slide.shapes.add_textbox(Inches(0.65), Inches(0.62), Inches(11.9), Inches(0.75))
    p = tb.text_frame.paragraphs[0]
    p.text = title
    p.font.name = "Aptos Display"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = INK


def add_footer(slide, number):
    tb = slide.shapes.add_textbox(Inches(11.8), Inches(7.08), Inches(0.9), Inches(0.22))
    p = tb.text_frame.paragraphs[0]
    p.text = f"{number:02d}"
    p.font.name = "Aptos"
    p.font.size = Pt(9)
    p.font.color.rgb = MUTED
    p.alignment = PP_ALIGN.RIGHT


def add_text(slide, x, y, w, h, text, size=20, color=INK, bold=False, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(0.03)
    tf.margin_right = Inches(0.03)
    p = tf.paragraphs[0]
    p.text = text
    p.font.name = "Aptos"
    p.font.size = Pt(size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.alignment = align
    return tb


def add_bullets(slide, x, y, w, h, bullets, size=18, color=INK):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(0.05)
    tf.margin_right = Inches(0.05)
    for i, text in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = text
        p.font.name = "Aptos"
        p.font.size = Pt(size)
        p.font.color.rgb = color
        p.space_after = Pt(12)
        p.level = 0
        p.text = "• " + p.text
    return tb


def add_card(slide, x, y, w, h, title, body, accent=BLUE):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = PALE
    shape.line.color.rgb = RGBColor(220, 228, 233)
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(0.08), Inches(h))
    bar.fill.solid()
    bar.fill.fore_color.rgb = accent
    bar.line.fill.background()
    add_text(slide, x + 0.22, y + 0.18, w - 0.4, 0.36, title, 16, accent, True)
    add_text(slide, x + 0.22, y + 0.62, w - 0.4, h - 0.75, body, 15, INK)


def add_image_crop(slide, page_no, box, x, y, w, h):
    src = f"pdf_extraido/page_{page_no:02d}.png"
    os.makedirs("slides_assets", exist_ok=True)
    img = Image.open(src)
    cropped = img.crop(box)
    path = f"slides_assets/crop_p{page_no}_{box[1]}_{box[3]}.png"
    cropped.save(path)
    slide.shapes.add_picture(path, Inches(x), Inches(y), width=Inches(w), height=Inches(h))
    return path


def new_slide(kicker, title, number):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide)
    add_header(slide, title, kicker)
    add_footer(slide, number)
    return slide


# 1
s = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(s, INK)
shape = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(0.18), prs.slide_height)
shape.fill.solid(); shape.fill.fore_color.rgb = ORANGE; shape.line.fill.background()
add_text(s, 0.85, 1.25, 11.6, 1.7,
         "Uma Análise Geométrica da\nHipótese do Manifold", 34, WHITE, True)
add_text(s, 0.88, 3.25, 11.2, 0.55,
         "Separabilidade linear em redes neurais feedforward", 20, RGBColor(205, 220, 228))
add_text(s, 0.88, 5.78, 9.5, 0.45,
         "Pedro Fernandes Galindo · Deoclecio Ivo de Melo Netto", 16, WHITE)
add_text(s, 0.88, 6.25, 9.5, 0.36,
         "Centro de Informática — UFPE", 14, RGBColor(205, 220, 228))
add_footer(s, 1)

# 2
s = new_slide("Contexto", "Por que analisar o interior da rede?", 2)
add_card(s, 0.65, 1.55, 3.75, 4.65, "CAIXA-PRETA",
         "MLPs classificam dados profundamente entrelaçados, mas a transformação interna costuma ser pouco explicada.")
add_card(s, 4.79, 1.55, 3.75, 4.65, "HIPÓTESE",
         "Dados reais se concentram perto de variedades de baixa dimensão. Redes aprendem a desemaranhar essas variedades.",
         ORANGE)
add_card(s, 8.93, 1.55, 3.75, 4.65, "PERGUNTA",
         "Como, exatamente, as camadas geométricas transformam os dados até torná-los linearmente separáveis?",
         BLUE)

# 3
s = new_slide("Ideia central", "A rede desenrola uma representação complicada", 3)
add_text(s, 0.75, 1.55, 3.1, 0.5, "ENTRADA", 16, BLUE, True)
add_text(s, 5.05, 1.55, 3.1, 0.5, "CAMADAS OCULTAS", 16, ORANGE, True)
add_text(s, 9.6, 1.55, 2.8, 0.5, "SAÍDA", 16, BLUE, True)
spiral = s.shapes.add_shape(MSO_SHAPE.SUN, Inches(0.75), Inches(2.2), Inches(3.2), Inches(3.2))
spiral.fill.solid(); spiral.fill.fore_color.rgb = LIGHT; spiral.line.color.rgb = BLUE
mid = s.shapes.add_shape(MSO_SHAPE.PENTAGON, Inches(5.05), Inches(2.2), Inches(3.2), Inches(3.2))
mid.fill.solid(); mid.fill.fore_color.rgb = RGBColor(252, 236, 214); mid.line.color.rgb = ORANGE
out = s.shapes.add_shape(MSO_SHAPE.CAN, Inches(9.55), Inches(2.45), Inches(3.2), Inches(2.7))
out.fill.solid(); out.fill.fore_color.rgb = LIGHT; out.line.color.rgb = BLUE
add_text(s, 1.15, 3.15, 2.4, 0.8, "duas classes\nentrelaçadas", 18, INK, True, PP_ALIGN.CENTER)
add_text(s, 5.43, 3.15, 2.45, 0.8, "esticar · dobrar\n· projetar", 18, INK, True, PP_ALIGN.CENTER)
add_text(s, 9.95, 3.15, 2.4, 0.8, "fronteira\nlinear", 18, INK, True, PP_ALIGN.CENTER)
add_text(s, 4.2, 3.35, 0.6, 0.4, "→", 28, MUTED, True, PP_ALIGN.CENTER)
add_text(s, 8.67, 3.35, 0.6, 0.4, "→", 28, MUTED, True, PP_ALIGN.CENTER)

# 4
s = new_slide("Formalização", "Cada camada é uma transformação do espaço", 4)
add_card(s, 0.65, 1.55, 5.85, 2.05, "PREFORMAÇÃO",
         "z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾", BLUE)
add_card(s, 6.83, 1.55, 5.85, 2.05, "ATIVAÇÃO",
         "a⁽ˡ⁾ = σ(z⁽ˡ⁾)", ORANGE)
add_card(s, 0.65, 3.95, 5.85, 2.05, "SAÍDA BINÁRIA",
         "ŷ = 1 / (1 + e⁻ᶻ⁽ᶫ⁾)", BLUE)
add_card(s, 6.83, 3.95, 5.85, 2.05, "REDE COMPLETA",
         "F(x) = f⁽ᶫ⁾ ◦ f⁽ᶫ⁻¹⁾ ◦ … ◦ f⁽¹⁾(x)", ORANGE)

# 5
s = new_slide("Treinamento", "A otimização aprende a transformação", 5)
add_bullets(s, 0.75, 1.55, 5.6, 4.8, [
    "Entropia cruzada binária",
    "Adam sobre lotes de 64 exemplos",
    "300 épocas na configuração principal",
    "Retropropagação: δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾ᵀδ⁽ˡ⁺¹⁾) ⊙ σ'(z⁽ˡ⁾)",
    "Taxa de aprendizado: η"
], 17)
add_card(s, 6.75, 1.55, 5.85, 4.8, "IMPLEMENTAÇÃO",
         "MLP implementada em NumPy\n\nPesos: inicialização de Glorot\nVieses: zero\nSaída: sigmoide\nAtivações: tanh e ReLU",
         ORANGE)

# 6
s = new_slide("Metodologia", "Um protocolo pequeno, controlado e reprodutível", 6)
add_card(s, 0.65, 1.55, 3.75, 2.05, "DADOS",
         "800 treino · 200 validação · 400 teste\nEspirais, luas e círculos")
add_card(s, 4.79, 1.55, 3.75, 2.05, "ARQUITETURA",
         "Principal: 3 × 8\nVarreduras: largura 2–16 · profundidade 1–4")
add_card(s, 8.93, 1.55, 3.75, 2.05, "OTIMIZAÇÃO",
         "lr: 0,001–0,1\nTaxa escolhida pela validação na grade")
add_card(s, 0.65, 4.05, 5.85, 2.05, "CONVERGIU?",
         "Acurácia de validação ≥ 0,95")
add_card(s, 6.83, 4.05, 5.85, 2.05, "CONTROLES",
         "Pesos aleatórios · 8 ou 512 neurônios · escalas de 1× a 30×",
         ORANGE)

# 7
s = new_slide("Medida", "Sondas: o quanto cada camada já separa as classes?", 7)
add_card(s, 0.65, 1.55, 5.85, 4.45, "SONDA LINEAR",
         "Treina uma regressão logística sobre as ativações da camada.\n\nA acurácia mede separabilidade linear; a rede original não é alterada.",
         BLUE)
add_card(s, 6.83, 1.55, 5.85, 4.45, "SONDA k-NN",
         "Avalia a pureza local das representações com k = 5.\n\nÉ uma medida menos exigente que uma fronteira linear.",
         ORANGE)
add_text(s, 0.75, 6.35, 11.8, 0.35,
         "Entrada → Oculta 1 → Oculta 2 → Oculta 3", 18, INK, True, PP_ALIGN.CENTER)

# 8
s = new_slide("Resultado principal", "Nas espirais, o ganho aparece nas camadas finais", 8)
add_image_crop(s, 6, (105, 690, 1170, 1300), 0.65, 1.55, 7.15, 4.55)
add_card(s, 8.05, 1.55, 4.63, 2.05, "CONFIGURAÇÃO",
         "tanh · 3 × 8 · lr = 0,01\n23 de 30 sementes convergem")
add_card(s, 8.05, 3.85, 4.63, 2.05, "AUMENTO",
         "Entrada: 0,61\nÚltima oculta: 0,99",
         ORANGE)

# 9
s = new_slide("Geometria", "A fronteira linear acompanha o desdobramento", 9)
add_image_crop(s, 7, (105, 720, 1165, 1160), 0.55, 1.55, 8.1, 3.75)
add_card(s, 8.93, 1.55, 3.75, 4.85, "LEITURA",
         "O eixo horizontal representa a direção da sonda linear.\n\nA linha vertical é sua fronteira de decisão.\n\nA projeção explica a separação linear, mas não toda a geometria em 8 dimensões.",
         ORANGE)

# 10
s = new_slide("Otimização", "A taxa de aprendizado muda a conclusão", 10)
add_image_crop(s, 8, (95, 55, 1180, 480), 0.55, 1.55, 7.1, 3.2)
add_card(s, 7.95, 1.55, 4.73, 2.0, "tanh",
         "lr 0,001 → apenas 3% convergem\nlr 0,03 → 97% convergem")
add_card(s, 7.95, 3.82, 4.73, 2.0, "ReLU",
         "Robusta entre 0,001 e 0,03\nQueda severa em 0,1",
         ORANGE)

# 11
s = new_slide("Arquitetura", "Largura pode ser mais decisiva que profundidade", 11)
add_image_crop(s, 8, (100, 790, 1180, 1430), 0.55, 1.55, 7.25, 4.75)
add_card(s, 8.05, 1.55, 4.63, 2.05, "2 neurônios",
         "Insuficientes em todas as profundidades\nMáximo: 0,68")
add_card(s, 8.05, 3.85, 4.63, 2.05, "16 neurônios",
         "ReLU: 0,99 com uma única camada\nComparação entre ativações: praticamente empate",
         ORANGE)

# 12
s = new_slide("Generalização", "O resultado muda conforme os dados e o ruído", 12)
add_image_crop(s, 9, (105, 420, 1165, 835), 0.55, 1.55, 7.25, 3.75)
add_card(s, 8.05, 1.55, 4.63, 2.05, "LUAS",
         "A acurácia sobe para 0,97 já na primeira ou segunda camada.")
add_card(s, 8.05, 3.85, 4.63, 2.05, "RUÍDO ALTO",
         "0,15: 0,71–0,76\n0,25: 0,59\nO limite vira a sobreposição das classes.",
         ORANGE)

# 13
s = new_slide("Controles", "Treinar não é a única forma de obter separabilidade", 13)
add_card(s, 0.65, 1.55, 3.75, 4.75, "CÍRCULOS",
         "Entrada: 0,48\n\n2 neurônios: 0,89\n3 neurônios: 1,00\n4 neurônios: 1,00",
         BLUE)
add_card(s, 4.79, 1.55, 3.75, 4.75, "REDE TREINADA",
         "tanh estreita: 0,99 na última camada\n\nReLU: 0,99")
add_card(s, 8.93, 1.55, 3.75, 4.75, "REDE ALEATÓRIA",
         "tanh · 512 neurônios · escala ≥3×\n\n0,99 na última camada, sem treino",
         ORANGE)

# 14
s = new_slide("Síntese", "Cinco conclusões bem estabelecidas neste experimento", 14)
add_bullets(s, 0.75, 1.55, 11.8, 5.1, [
    "A separabilidade linear cresce ao longo das camadas nas redes convergidas.",
    "O ganho é concentrado nas camadas intermediárias/finais e nem sempre monotônico.",
    "Largura 2 é insuficiente; redes 2D circundadas exigem pelo menos 3 neurônios.",
    "Diferenças tanh × ReLU dependem fortemente da taxa de aprendizado.",
    "Uma projeção aleatória ampla pode separar dados sem nenhum treinamento."
], 19)

# 15
s = new_slide("Discussão", "O que os resultados explicam — e o que não explicam", 15)
add_card(s, 0.65, 1.55, 5.85, 4.65, "EVIDÊNCIA A FAVOR",
         "Compatível com o desdobramento geométrico da variedade.\n\nMostra onde a separabilidade aumenta e como arquitetura/otimização influenciam.")
add_card(s, 6.83, 1.55, 5.85, 4.65, "LIMITAÇÃO",
         "Dados sintéticos 2D e redes pequenas.\n\nSondas medem separabilidade, não propriedades diretas do manifold.\n\nTaxas, épocas e hiperparâmetros limitados.",
         ORANGE)

# 16
s = new_slide("Conclusão", "Treino, arquitetura e geometria precisam ser lidos juntos", 16)
add_text(s, 0.75, 1.65, 11.7, 1.15,
         "Redes que convergem transformam os dados até que uma\nfronteira linear capture a separação das classes.", 27, INK, True, PP_ALIGN.CENTER)
add_card(s, 0.65, 3.35, 3.75, 2.15, "OTIMIZAÇÃO",
         "A taxa pode dominar a aparente vantagem entre ativações.")
add_card(s, 4.79, 3.35, 3.75, 2.15, "ARQUITETURA",
         "Largura e profundidade têm limites diferentes.")
add_card(s, 8.93, 3.35, 3.75, 2.15, "EXPANSÃO",
         "Dados reais, dimensão intrínseca e homologia persistente.",
         ORANGE)

prs.save(OUT)

script = """# Roteiro sugerido — 15 a 20 minutos

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
"""

open(NOTES, "w", encoding="utf-8").write(script)
print(OUT)
print(NOTES)
