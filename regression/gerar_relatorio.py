# Gera o relatório em PDF (relatorio.pdf) com as tabelas e gráficos exigidos
# pelo enunciado, a partir dos arquivos salvos em ./resultados pelas etapas 1 a 6.

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from config import PASTA_RESULTADOS

ARQ_PDF = "relatorio.pdf"

pdfmetrics.registerFont(TTFont('Arial', r'C:\Windows\Fonts\arial.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Bold', r'C:\Windows\Fonts\arialbd.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Italic', r'C:\Windows\Fonts\ariali.ttf'))
pdfmetrics.registerFont(TTFont('Arial-BoldItalic', r'C:\Windows\Fonts\arialbi.ttf'))
pdfmetrics.registerFontFamily('Arial', normal='Arial', bold='Arial-Bold',
                              italic='Arial-Italic', boldItalic='Arial-BoldItalic')

# Cores dos cabeçalhos, inspiradas no modelo do enunciado
AZUL = colors.HexColor('#DEEAF6')
LARANJA_CLARO = colors.HexColor('#FBE4D5')
LARANJA = colors.HexColor('#ED7D31')
LILAS = colors.HexColor('#B4A7D6')

ESTILO_TITULO = ParagraphStyle('titulo', fontName='Arial-Bold', fontSize=16,
                               leading=20, spaceAfter=14)
ESTILO_SECAO = ParagraphStyle('secao', fontName='Arial-Bold', fontSize=11,
                              leading=14, spaceBefore=12, spaceAfter=4)
ESTILO_SUB = ParagraphStyle('sub', fontName='Arial', fontSize=9, leading=11,
                            spaceAfter=4)
ESTILO_CELULA = ParagraphStyle('cel', fontName='Arial', fontSize=9, leading=11,
                               alignment=TA_CENTER)
ESTILO_CELULA_ESQ = ParagraphStyle('cel_esq', fontName='Arial', fontSize=9, leading=11)


def ler(arquivo):
    return pd.read_csv(PASTA_RESULTADOS / arquivo, index_col=0, dtype=str,
                       keep_default_na=False)


def virgula(v):
    """Troca o ponto decimal por vírgula em valores numéricos."""
    try:
        float(v)
        return str(v).replace('.', ',')
    except ValueError:
        return str(v)


def tabela(cabecalho, linhas, larguras, cor_cab, texto_cab=colors.black):
    cor_hex = texto_cab.hexval()[2:]
    dados = [[Paragraph(f'<font color="#{cor_hex}"><b>{c}</b></font>',
                        ESTILO_CELULA if i else ESTILO_CELULA_ESQ)
              for i, c in enumerate(cabecalho)]]
    for linha in linhas:
        dados.append([Paragraph(str(c), ESTILO_CELULA if i else ESTILO_CELULA_ESQ)
                      for i, c in enumerate(linha)])
    t = Table(dados, colWidths=larguras)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), cor_cab),
        ('TEXTCOLOR', (0, 0), (-1, 0), texto_cab),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    t.hAlign = 'LEFT'
    return t


def secao(titulo, *conteudo):
    return KeepTogether([Paragraph(titulo, ESTILO_SECAO), *conteudo])


def linhas_resultados(df):
    nomes = {'TREINO MSE (dpa)': 'TREINO MSE médio (dpa)',
             'VALIDAÇÃO MSE (dpa)': 'VALIDAÇÃO MSE médio (dpa)',
             'MÉDIA DAS DIFS. (dpa)': 'MÉDIA DAS DIFS. (dpa)'}
    return [[nomes.get(i, i)] + list(df.loc[i]) for i in df.index]


def main():
    story = [Paragraph("Tarefa 2: Regressão", ESTILO_TITULO)]

    # 1) Dataset de treinamento/validação
    t1 = ler("etapa1_dataset.csv")
    v = t1['VALOR']
    linhas1 = [
        ["Vítimas tri=verde", v['Vítimas tri=verde']],
        ["Vítimas tri=amarelo", v['Vítimas tri=amarelo']],
        ["Vítimas tri=vermelho", v['Vítimas tri=vermelho']],
        ["Vítimas tri=preto", v['Vítimas tri=preto']],
        ["Idade das vítimas", virgula(v['Idade das vítimas (média amostral)'])],
        ["Desvio padrão amostral da idade", virgula(v['Desvio padrão amostral da idade'])],
        ["ruído", virgula(v['ruído'])],
        ["Número de folds (k)", v['Número de folds (validação cruzada)']],
    ]
    story.append(secao("1) DATASET DE TREINAMENTO/VALIDAÇÃO",
                       tabela(["MODELO", "VALOR"], linhas1, [6.5 * cm, 4 * cm], AZUL)))

    # 2) Hiperparametrizações CART
    t2 = ler("etapa2_cart_hiperparametros.csv")
    ordem2 = ['Min_samples_leaf', 'Max_depth', 'Min_samples_split', 'Criterion']
    linhas2 = [[i] + list(t2.loc[i]) for i in ordem2]
    story.append(secao("2) HIPERPARAMETRIZAÇÕES CART",
                       tabela(["MODELO", "U", "E", "O"], linhas2,
                              [4.5 * cm, 3.5 * cm, 3.5 * cm, 3.5 * cm], AZUL)))

    # 3) Resultados CART
    t3 = ler("etapa2_cart_resultados.csv")
    story.append(secao(
        "3) RESULTADOS DOS MODELOS CART",
        Paragraph("<i>Médias dos MSEs com desvio padrão amostral por fold por "
                  "hiperparametrização (U, E, O)</i>", ESTILO_SUB),
        tabela(["MODELO"] + list(t3.columns), linhas_resultados(t3),
               [5 * cm, 3.8 * cm, 3.8 * cm, 3.8 * cm], AZUL)))

    # 4) Hiperparametrizações RN
    t4 = ler("etapa3_rn_hiperparametros.csv")
    linhas4 = [[i] + [virgula(x) for x in t4.loc[i]] for i in t4.index]
    story.append(secao("4) HIPERPARAMETRIZAÇÕES REDE NEURAL",
                       tabela(["MODELO", "U", "E", "O"], linhas4,
                              [4.5 * cm, 3.8 * cm, 3.8 * cm, 3.8 * cm], LARANJA_CLARO)))

    # 5) Resultados RN
    t5 = ler("etapa3_rn_resultados.csv")
    story.append(secao("5) RESULTADOS DOS MODELOS RN",
                       tabela(["MODELO"] + list(t5.columns), linhas_resultados(t5),
                              [5 * cm, 3.8 * cm, 3.8 * cm, 3.8 * cm], LARANJA_CLARO)))

    # 6) Melhor CART x melhor RN
    t6 = ler("etapa4_comparacao.csv")
    story.append(secao("6) MELHOR MODELO CART X MELHOR MODELO RN",
                       tabela(["MODELO"] + list(t6.columns), linhas_resultados(t6),
                              [5 * cm, 4 * cm, 4 * cm], LARANJA, colors.white)))

    # 7) Teste cego
    t7 = ler("etapa6_teste_cego.csv")
    linhas7 = [[i] + list(t7.loc[i]) for i in t7.index]
    story.append(secao("7) RESULTADOS DO TESTE CEGO: MELHOR CART X MELHOR RN",
                       tabela(["MODELO"] + list(t7.columns), linhas7,
                              [5 * cm, 4 * cm, 4 * cm], LILAS)))

    # 8) Gráficos de dispersão
    lado = 8.2 * cm
    graficos = Table([[Image(str(PASTA_RESULTADOS / "etapa6_dispersao_cart.png"), lado, lado * 5 / 5.2),
                       Image(str(PASTA_RESULTADOS / "etapa6_dispersao_rn.png"), lado, lado * 5 / 5.2)]],
                     colWidths=[lado + 0.3 * cm, lado + 0.3 * cm])
    graficos.hAlign = 'LEFT'
    story.append(secao("8) GRÁFICO DE DISPERSÃO", graficos))

    doc = SimpleDocTemplate(ARQ_PDF, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="Tarefa 2: Regressão")
    doc.build(story)
    print(f"Relatório salvo em {ARQ_PDF}")


if __name__ == "__main__":
    main()
