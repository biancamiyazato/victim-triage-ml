# Gera o relatório em PDF (relatorio.pdf) com as tabelas 1 a 8 do enunciado,
# a partir dos resultados salvos pelas etapas 1 a 5.

import json

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, KeepTogether

from config import TREINO_DIR, RESULTADOS_DIR, CLASSES

SAIDA_PDF = "relatorio.pdf"


def carregar(caminho):
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


dataset = carregar(TREINO_DIR / "resumo_dataset.json")
cart = carregar(RESULTADOS_DIR / "etapa2_cart.json")
rn = carregar(RESULTADOS_DIR / "etapa3_rn.json")
melhores = carregar(RESULTADOS_DIR / "etapa4_melhores.json")
teste = carregar(RESULTADOS_DIR / "etapa5_teste_cego.json")

estilos = getSampleStyleSheet()
TITULO = ParagraphStyle("titulo", parent=estilos["Title"], fontSize=18, spaceAfter=4)
SUBTITULO = ParagraphStyle("sub", parent=estilos["Normal"], fontSize=10, alignment=1,
                           textColor=colors.HexColor("#555555"), spaceAfter=10)
SECAO = ParagraphStyle("secao", parent=estilos["Heading4"], fontSize=10.5, spaceBefore=12, spaceAfter=4)
CELULA = ParagraphStyle("celula", parent=estilos["Normal"], fontSize=9, leading=11)
CELULA_C = ParagraphStyle("celula_c", parent=CELULA, alignment=1)
CELULA_B = ParagraphStyle("celula_b", parent=CELULA_C, fontName="Helvetica-Bold")
MONO = ParagraphStyle("mono", parent=CELULA_C, fontName="Courier")

COR_CART = colors.HexColor("#DAEEF3")
COR_RN = colors.HexColor("#FCE4D6")
COR_DATASET = colors.HexColor("#E4DFEC")
COR_MELHOR = colors.HexColor("#B4A7D6")

F1 = "<i>f</i><sub>1</sub> médio"


def tabela(linhas, larguras, cor_cabecalho, primeira_coluna_esq=True):
    dados = []
    for i, linha in enumerate(linhas):
        celulas = []
        for j, valor in enumerate(linha):
            if i == 0:
                estilo = CELULA_B
            elif j == 0 and primeira_coluna_esq:
                estilo = CELULA
            else:
                estilo = CELULA_C
            celulas.append(Paragraph(str(valor), estilo))
        dados.append(celulas)
    t = Table(dados, colWidths=[w * cm for w in larguras])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), cor_cabecalho),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#808080")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    t.hAlign = "LEFT"
    return t


def fmt(r):
    return f"{r['media']:.5f} <font size=7.5>({r['dpa']:.5f})</font>"


def linhas_resultados(resultados):
    return [
        [f"TREINO {F1} (DPA)"] + [fmt(r["treino"]) for r in resultados],
        [f"VALIDAÇÃO {F1} (DPA)"] + [fmt(r["validacao"]) for r in resultados],
        ["MÉDIA DAS DIFS. (DPA)"] + [fmt(r["difs"]) for r in resultados],
    ]


def secao(titulo, conteudo):
    return KeepTogether([Paragraph(titulo, SECAO), conteudo])


story = [
    Paragraph("Tarefa 1: Classificação", TITULO),
    Paragraph("UTFPR/Curitiba - Sistemas Inteligentes 1 - 2026/2", SUBTITULO),
]

# 1) Dataset de treinamento/validação
por_tri = dataset["vitimas_por_tri"]
story.append(secao("1) DATASET DE TREINAMENTO/VALIDAÇÃO", tabela(
    [["MODELO", "VALOR"]]
    + [[f"Vítimas tri={nome}", por_tri[nome]] for nome in CLASSES.values()]
    + [["Idade das vítimas", f"{dataset['idade_media_amostral']:.2f}"],
       ["Desvio padrão amostral da idade", f"{dataset['idade_dpa']:.2f}"],
       ["ruído", dataset["parametros_gerador"]["nivel_ruido"]],
       ["Tipo de acidente", "não se aplica: o gerador não modela tipo de acidente"],
       ["Número de folds", dataset["n_folds"]]],
    [6, 8], COR_DATASET)))

# 2) Hiperparametrizações CART
hp = cart["hiperparametros"]
modelos = ["U", "E", "O"]
story.append(secao("2) HIPERPARAMETRIZAÇÕES CART", tabela(
    [["MODELO"] + modelos,
     ["Min_samples_leaf"] + [hp[m]["min_samples_leaf"] for m in modelos],
     ["Max_depth"] + [hp[m]["max_depth"] if hp[m]["max_depth"] is not None else "None" for m in modelos],
     ["Criterion (gini, ent)"] + [hp[m]["criterion"] for m in modelos]],
    [4.5, 3.5, 3.5, 3.5], COR_CART)))

# 3) Resultados CART
story.append(secao("3) RESULTADOS DOS MODELOS CART", tabela(
    [["MODELO"] + modelos] + linhas_resultados([cart["resultados"][m] for m in modelos]),
    [4.5, 4, 4, 4], COR_CART)))

# 4) Hiperparametrizações RN
hp = rn["hiperparametros"]


def opcional(m, chave, padrao):
    return hp[m].get(chave, padrao)


story.append(secao("4) HIPERPARAMETRIZAÇÕES REDE NEURAL", tabela(
    [["MODELO"] + modelos,
     ["Topologia"] + ["[" + " ".join(map(str, hp[m]["hidden_layer_sizes"])) + "]" for m in modelos],
     ["Função de ativação"] + [hp[m]["activation"] for m in modelos],
     ["Learning rate"] + [hp[m]["learning_rate_init"] for m in modelos],
     ["solver"] + [hp[m]["solver"] for m in modelos],
     ["alpha (regularização L2)"] + [hp[m]["alpha"] for m in modelos],
     ["max_iter"] + [hp[m]["max_iter"] for m in modelos],
     ["early_stopping"] + [hp[m]["early_stopping"] for m in modelos],
     ["tol"] + [opcional(m, "tol", 0.0001) for m in modelos],
     ["n_iter_no_change"] + [opcional(m, "n_iter_no_change", 10) for m in modelos],
     ["Pré-processamento"] + [rn["preprocessamento"]] * 3],
    [4.5, 4, 4, 4], COR_RN)))

# 5) Resultados RN
story.append(secao("5) RESULTADOS DOS MODELOS RN", tabela(
    [["MODELO"] + modelos] + linhas_resultados([rn["resultados"][m] for m in modelos]),
    [4.5, 4, 4, 4], COR_RN)))

# 6) Melhor CART x melhor RN
m_cart, m_rn = melhores["cart"]["modelo"], melhores["rn"]["modelo"]
story.append(secao("6) MELHOR MODELO CART X MELHOR MODELO RN", tabela(
    [["MODELO", f"Melhor CART ({m_cart})", f"Melhor RN ({m_rn})"]]
    + linhas_resultados([melhores["cart"]["resultados"], melhores["rn"]["resultados"]]),
    [4.5, 4, 4], COR_MELHOR)))

# 7) Teste cego
res_cart, res_rn = teste["resultados"].values()
story.append(secao(f"7) RESULTADOS DO TESTE CEGO MELHOR CART X MELHOR RN ({teste['n_vitimas_teste']} vítimas)", tabela(
    [["MODELO", f"Melhor CART ({m_cart})", f"Melhor RN ({m_rn})"]]
    + [[rotulo, f"{res_cart[chave]:.5f}", f"{res_rn[chave]:.5f}"] for chave, rotulo in [
        ("precisao_macro", "MÉDIA PRECISÃO (macro)"), ("recall_macro", "MÉDIA RECALL (macro)"),
        ("f1_macro", "F1 SCORE (macro)"), ("acuracia", "ACURÁCIA")]],
    [4.5, 4, 4], COR_MELHOR)))

# 8) Matrizes de confusão
siglas = ["G", "Y", "R", "B"]


def matriz(mc):
    linhas = [["", ""] + siglas]
    for i, sigla in enumerate(siglas):
        linhas.append(["<i>real</i>" if i == 1 else "", sigla] + mc[i])
    linhas.append(["", "", "<i>predito</i>", "", "", ""])
    dados = [[Paragraph(str(v), MONO) for v in linha] for linha in linhas]
    t = Table(dados, colWidths=[1.5 * cm, 0.7 * cm] + [1.1 * cm] * 4)
    t.setStyle(TableStyle([
        ("SPAN", (2, 5), (5, 5)),
        ("LINEBELOW", (2, 0), (5, 0), 0.5, colors.grey),
        ("LINEAFTER", (1, 1), (1, 4), 0.5, colors.grey),
        ("BACKGROUND", (2, 1), (2, 1), colors.HexColor("#EEEEEE")),
        ("BACKGROUND", (3, 2), (3, 2), colors.HexColor("#EEEEEE")),
        ("BACKGROUND", (4, 3), (4, 3), colors.HexColor("#EEEEEE")),
        ("BACKGROUND", (5, 4), (5, 4), colors.HexColor("#EEEEEE")),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
    ]))
    return t


matrizes = Table(
    [[Paragraph(f"CART ({m_cart})", CELULA_B), Paragraph(f"RN ({m_rn})", CELULA_B)],
     [matriz(res_cart["matriz_confusao"]), matriz(res_rn["matriz_confusao"])]],
    colWidths=[7.5 * cm, 7.5 * cm])
matrizes.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), COR_MELHOR),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#808080")),
    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
]))
matrizes.hAlign = "LEFT"
story.append(secao("8) MATRIZ DE CONFUSÃO DO MELHOR CART e MATRIZ DE CONFUSÃO DA MELHOR RN<br/>"
                   "G=verde, Y=amarelo, R=vermelho, B=preto", matrizes))

doc = SimpleDocTemplate(SAIDA_PDF, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                        title="Tarefa 1: Classificação - Relatório")
doc.build(story)
print(f"Relatório salvo em {SAIDA_PDF}")
