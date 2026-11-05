# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:47:28 2026

@author: cesar

Gera um mapa do ambiente simulando a exploração realizada por um sistema de
agentes exploradores.
             
Configurações: 
    pExplored: probabilidade de uma posição qualquer ter sido explorada; caso
               não tenha sido não é incluída no arquivo de saída
    ocultar_tri_sobr: caso True, não coloca o valor de triagem e de prob. de
             sobrevivência no arquivo de saída. Uso: tarefas para estudantes
             que devem usar um classificador ou um regressor.

Entrada:
    arquivos de configuração do ambiente: env_config.txt, env_obst.txt e 
    env_victims.txt
    
    arquivos de sinais vitais das vitimas: data.csv
    
    
Saída:
    - map.csv:
    Representa o mapa explorado. Cada linha contém:

        x_rel, y_rel, obst, id, <sinais vitais>

    onde:
        x_rel = x - BASE.x
        y_rel = y - BASE.y
        obst  = dificuldade da célula (ou 1 se não houver obstáculo)
        id   = id da vítima ou -1 se não houver
        sinais vitais:
              valores do data.csv ou -1 se não houver vítima

    Observação:
    Nem todas as células são incluídas no arquivo, conforme a probabilidade
    pExplored (simulação de exploração parcial).
"""

import pygame
import csv
import random

# =========================
# CONFIGURAÇÕES
# =========================
path_config = "./datasets/env/94x94_408v/"
path_vict = "./datasets/vict/408v/"
path_map = "./map/"
pExplored = 1.0
ocultar_tri_sobr = False

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
CYAN = (0, 255, 255)

TRI_COLORS = {
    -1: (120, 120, 120), # tri desconhecido
    0: (0, 255, 0),     # verde
    1: (255, 255, 0),   # amarelo
    2: (255, 0, 0),     # vermelho
    3: (0, 0, 0)        # preto
}

# =========================
# LEITURA CONFIG
# =========================
def read_config():
    with open(path_config + "env_config.txt") as f:
        lines = f.readlines()
    
    prefixo, coordenadas = lines[0].split(maxsplit=1)   # separa "BASE" do resto
    base_x, base_y = map(int, coordenadas.replace(" ", "").split(",")) # elimina espaços em 5,  10 =>5,10
    #base_x, base_y = map(int, base_line[1].split(","))
    W = int(lines[1].split()[1])   # grid_width
    H = int(lines[2].split()[1])   # grid_height
    WINW = int(lines[3].split()[1])   # WINDOW width
    WINH = int(lines[4].split()[1])   # WINDOW heigth
    CELLW = WINW/W
    CELLH = WINH/H
    return base_x, base_y, W, H, CELLW, CELLH


# =========================
# LEITURA OBSTÁCULOS
# =========================
def read_obstacles():
    obst = {}
    with open(path_config + "env_obst.txt") as f:
        reader = csv.reader(f)
        for row in reader:
            x, y, obst_dif = int(row[0]), int(row[1]), float(row[2])
            obst[(x, y)] = obst_dif

    return obst


# =========================
# LEITURA VÍTIMAS
# =========================
def read_victims():
    victims = {}
    with open(path_config + "env_victims.txt") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            x, y = int(row[0]), int(row[1])
            victims[(x, y)] = i
 
    return victims


# =========================
# LEITURA SINAIS VITAIS
# =========================
def read_vital_signals():
    with open(path_vict + "data.csv") as f:
        reader = csv.reader(f)
        header = next(reader)
        data = list(reader)
    return header, data


# =========================
# GERA MAPA
# =========================
def generate_map():
    base_x, base_y, W, H, CELLW, CELLH = read_config()
    obst = read_obstacles()
    victims = read_victims()
    header, vital_data = read_vital_signals()

    with open(path_map + "map.csv", "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["x_rel", "y_rel", "obst", "id"] + header)

        for x in range(W):
            for y in range(H):

                if random.random() > pExplored:
                    continue
 
                x_rel = x - base_x
                y_rel = y - base_y

                obst_dif = obst.get((x, y), 1)
                id = victims.get((x, y), -1)

                if id != -1:
                    vitals = vital_data[id]

                    if ocultar_tri_sobr:
                        idx_tri = header.index("tri")
                        idx_ps = header.index("sobr")
                        vitals[idx_tri] = "-1"
                        vitals[idx_ps] = "-1"
                else:
                    vitals = ["-1"] * len(header)

                writer.writerow([x_rel, y_rel, obst_dif, id] + vitals)


# =========================
# PLOT
# =========================
def plot_map():

    pygame.init()

    base_x, base_y, W, H, CELLW, CELLH = read_config()
    header, _ = read_vital_signals()

    screen = pygame.display.set_mode((W * CELLW, H * CELLH))
    pygame.display.set_caption("Map Inspector")

    cell_data = {}

    with open(path_map + "map.csv") as f:
        reader = csv.reader(f)
        header_map = next(reader)

        for row in reader:
            x_rel, y_rel = int(row[0]), int(row[1])
            x = x_rel + base_x
            y = y_rel + base_y

            cell_data[(x, y)] = row

    selected_cell = None

    running = True
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', int(CELLW * 0.6), bold=True)
    while running:

        screen.fill(WHITE)

        # --- DRAW CELLS ---
        for x in range(W):
            for y in range(H):
                rect = (x * CELLW, y * CELLH, CELLW, CELLH)

                if (x, y) in cell_data:
                    row = cell_data[(x, y)]
                    obst_dif = float(row[2])
                    
                    if obst_dif == 100.0:
                        color = (0, 0, 0) ## black
                    else:
                        gray = int(255 - (obst_dif / 6) * 255)
                        color = (200, gray, gray)
                        
                    pygame.draw.rect(screen, color, rect)
                else:  
                    # célula não explorada
                    pygame.draw.rect(screen, (240, 240, 240), rect)
        
                    # --- DESENHA "?" ---
                    text = font.render("?", True, (100, 100, 100))
                    text_rect = text.get_rect(
                        center=(x * CELLW + CELLW / 2,
                                y * CELLH + CELLH / 2)
                    )
                    screen.blit(text, text_rect)
                

        # --- BASE ---
        pygame.draw.rect(
            screen,
            CYAN,
            (base_x * CELLW, base_y * CELLH, CELLW, CELLH)
        )

        # --- DRAW VICTIMS ---
        tri_idx = header.index("tri")
        
        for (x, y), row in cell_data.items():
        
            id = int(row[3])

        
            if id != -1:
        
                tri_val = row[4 + tri_idx]
                tri = int(tri_val) if tri_val != "-1" else -1
        
                if tri in TRI_COLORS:
        
                    color = TRI_COLORS[tri]
        
                    pygame.draw.circle(
                        screen,
                        color,
                        (int(x * CELLW + CELLW / 2),
                         int(y * CELLH + CELLH / 2)),
                        CELLW // 3
                    )

        # --- GRID LINES ---
        for x in range(W):
            for y in range(H):
                pygame.draw.rect(
                    screen,
                    (160, 160, 160),
                    (x * CELLW, y * CELLH, CELLW, CELLH),
                    -1
                )

        # --- INSPECTOR ---
        if selected_cell:

            c, r = selected_cell

            pygame.draw.rect(
                screen,
                (0, 0, 255),
                (c * CELLW, r * CELLH, CELLW, CELLH),
                3
            )

            font = pygame.font.SysFont('Arial', 16)

            lines = [f"Cell: ({c},{r})"]

            if (c, r) in cell_data:
                row = cell_data[(c, r)]
                obst_dif = row[2]
                id = int(row[3])

                lines.append(f"Difficulty: {obst_dif}")

                if id != -1:
                    lines.append(f"Victim id: {id}")

                    for name, value in zip(header, row[4:]):
                        lines.append(f"{name}: {value}")
                else:
                    lines.append("No victim")
            else:
                lines.append("Unexplored")

            panel = pygame.Surface((260, 20 * len(lines)))
            panel.set_alpha(200)
            panel.fill((255, 255, 255))

            screen.blit(panel, (10, 10))

            for i, line in enumerate(lines):
                text = font.render(line, True, BLACK)
                screen.blit(text, (15, 15 + i * 20))

        pygame.display.flip()

        # --- EVENTS ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                px, py = event.pos
                selected_cell = (int(px / CELLW), int(py / CELLH))
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    selected_cell = None

        clock.tick(30)

    # =========================
    # CONTAGEM FINAL
    # =========================
    tri_count = {0: 0, 1: 0, 2: 0, 3: 0}
    tot_vict = 0

    for row in cell_data.values():
        id = int(row[3])
        if id != -1:
            tot_vict += 1
            tri = int(row[16]) if row[16] != "-1" else -1
            if tri in tri_count:
                tri_count[tri] += 1

    if ocultar_tri_sobr == False: 
        print("\nVictim count by tri:")
        for k, v in tri_count.items():
            print(f"tri={k}: {v}")

    print(f"Found victims: {tot_vict}")


    pygame.quit()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    generate_map()
    plot_map()