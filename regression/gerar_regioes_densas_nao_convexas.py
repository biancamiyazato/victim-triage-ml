# -*- coding: utf-8 -*-
"""
Created on Tue May 12 18:28:05 2026

@author: cesar
"""

"""
GERADOR DE DATASET ARTIFICIAL PARA DBSCAN
=========================================
OBJETIVOS:
---------
1. Gerar regiões densas NÃO-CONVEXAS:
   - meia-lua
   - anel
   - espiral

2. Espalhar regiões pelo grid inteiro:
   - macrozonas distribuídas
   - evitar concentração espacial

3. Permitir ENTRELAÇAMENTO:
   - regiões próximas
   - ex.: duas meias-luas próximas

4. Respeitar obstáculos:
   - env_obst.txt
   - dificuldade == 100 => célula proibida

5. Reorganizar vítimas:
   - data_input.csv -> data.csv
   - baseado em múltiplos intervalos de sobr

SAÍDAS:
-------
./datasets/env/94x94_408v/env_victims.txt
./datasets/vict/408v/data.csv
"""

import pandas as pd
import random
import math
import os


# =========================================================
# CONFIGURAÇÕES
# =========================================================
GRID_SIZE = 94

ARQ_OBST = "./datasets/env/94x94_408v/env_obst.txt"
ARQ_ENV_OUT = "./datasets/env/94x94_408v/env_victims.txt"

ARQ_DATA_INPUT = "./datasets/vict/408v/data_input.csv"
ARQ_DATA_OUT = "./datasets/vict/408v/data.csv"

SEED = 123
random.seed(SEED)


# =========================================================
# DEFINIÇÃO FLEXÍVEL DAS REGIÕES
# (
#   quantidade,
#   forma,
#   [(sobr_min, sobr_max), ...]
# )
# =========================================================
REGIONS = [
    (35, "anel", [(0.0, 0.1), (0.95, 1.00)]),
    (25, "anel", [(0.0, 0.1), (0.75, 0.85)]),
    (55, "meia_lua", [(0.60, 0.75)]),
    (80, "espiral", [(0.45, 0.60), (0.9, 1.0)]),
    (65, "anel", [(0.30, 0.45), (0.0, 0.05)]),
    (80, "meia_lua", [(0.15, 0.30)]),
    (68, "espiral", [(0.00, 0.15), (0.95,1.0)]),
]


# =========================================================
# VALIDAÇÃO DE VÍTIMAS
# =========================================================
df_input = pd.read_csv(ARQ_DATA_INPUT)

TOTAL_VICTIMS = len(df_input)

TOTAL_REQUIRED = sum(
    qtd for qtd, _, _ in REGIONS
)

if TOTAL_REQUIRED != TOTAL_VICTIMS:
    raise ValueError(
        f"Soma REGIONS={TOTAL_REQUIRED} "
        f"!= vítimas={TOTAL_VICTIMS}"
    )


# =========================================================
# CARREGAR OBSTÁCULOS
# =========================================================
obst = pd.read_csv(
    ARQ_OBST,
    header=None,
    names=["x", "y", "dificuldade"]
)

obst["x"] = obst["x"].astype(int)
obst["y"] = obst["y"].astype(int)
obst["dificuldade"] = obst["dificuldade"].astype(float)

blocked = set(
    zip(
        obst[obst["dificuldade"] == 100]["x"],
        obst[obst["dificuldade"] == 100]["y"]
    )
)


# =========================================================
# CÉLULAS LIVRES
# =========================================================
free_cells = set()

for x in range(GRID_SIZE):
    for y in range(GRID_SIZE):

        if (x, y) not in blocked:
            free_cells.add((x, y))


# =========================================================
# AUXILIARES
# =========================================================
def clamp_grid(x, y):
    return (
        0 <= x < GRID_SIZE and
        0 <= y < GRID_SIZE
    )


# =========================================================
# ZONAS GLOBAIS
# =========================================================
def build_global_anchor_zones(
    grid_size,
    num_regions
):
    zones = []

    divisions = 3

    step = grid_size // divisions

    for gx in range(divisions):
        for gy in range(divisions):

            x_min = gx * step
            x_max = min(
                grid_size - 1,
                (gx + 1) * step - 1
            )

            y_min = gy * step
            y_max = min(
                grid_size - 1,
                (gy + 1) * step - 1
            )

            zones.append(
                (
                    x_min,
                    x_max,
                    y_min,
                    y_max
                )
            )

    random.shuffle(zones)

    return zones[:num_regions]


# =========================================================
# CENTRO EM ZONA
# =========================================================
def choose_center_in_zone(
    available_cells,
    zone,
    margin=4
):
    x_min, x_max, y_min, y_max = zone

    candidates = []

    for x, y in available_cells:

        if (
            x_min + margin <= x <= x_max - margin and
            y_min + margin <= y <= y_max - margin
        ):
            candidates.append((x, y))

    if not candidates:
        raise ValueError(
            "Sem centro válido na zona."
        )

    return random.choice(candidates)


# =========================================================
# CENTRO ENTRELAÇADO
# =========================================================
def choose_interlaced_center(
    base_center,
    available_cells,
    max_offset=8
):
    bx, by = base_center

    candidates = []

    for dx in range(-max_offset, max_offset + 1):
        for dy in range(-max_offset, max_offset + 1):

            nx = bx + dx
            ny = by + dy

            if (
                clamp_grid(nx, ny) and
                (nx, ny) in available_cells
            ):

                d = math.sqrt(dx**2 + dy**2)

                if 2 <= d <= max_offset:
                    candidates.append((nx, ny))

    if not candidates:
        raise ValueError(
            "Sem centro entrelaçado."
        )

    return random.choice(candidates)


# =========================================================
# SHAPES
# =========================================================
def generate_ring_region(
    available_cells,
    cluster_size,
    center
):
    center_x, center_y = center

    radius = random.randint(4, 12)
    thickness = random.randint(2, 7)

    candidates = []

    for x in range(
        center_x - radius - thickness,
        center_x + radius + thickness + 1
    ):
        for y in range(
            center_y - radius - thickness,
            center_y + radius + thickness + 1
        ):

            if not clamp_grid(x, y):
                continue

            if (x, y) not in available_cells:
                continue

            d = math.sqrt(
                (x - center_x) ** 2 +
                (y - center_y) ** 2
            )

            if radius <= d <= radius + thickness:
                candidates.append((x, y))

    if len(candidates) < cluster_size:
        raise ValueError

    chosen = random.sample(
        candidates,
        cluster_size
    )

    for p in chosen:
        available_cells.remove(p)

    return chosen


def generate_crescent_region(
    available_cells,
    cluster_size,
    center
):
    center_x, center_y = center

    outer_r = random.randint(7, 12)
    inner_r = outer_r - random.randint(2, 3)

    angle_start = random.uniform(
        0,
        math.pi
    )

    angle_end = angle_start + random.uniform(
        math.pi * 0.8,
        math.pi * 1.3
    )

    offset_x = random.randint(2, 4)

    candidates = []

    for x in range(
        center_x - outer_r - 2,
        center_x + outer_r + 2
    ):
        for y in range(
            center_y - outer_r - 2,
            center_y + outer_r + 2
        ):

            if not clamp_grid(x, y):
                continue

            if (x, y) not in available_cells:
                continue

            dx = x - center_x
            dy = y - center_y

            angle = math.atan2(dy, dx)

            if angle < 0:
                angle += 2 * math.pi

            d_outer = math.sqrt(
                dx**2 + dy**2
            )

            d_inner = math.sqrt(
                (x - (center_x + offset_x))**2 +
                dy**2
            )

            if (
                angle_start <= angle <= angle_end and
                d_outer <= outer_r and
                d_inner >= inner_r
            ):
                candidates.append((x, y))

    if len(candidates) < cluster_size:
        raise ValueError

    chosen = random.sample(
        candidates,
        cluster_size
    )

    for p in chosen:
        if random.random() < 0.85:
            available_cells.remove(p)
       

    return chosen


def generate_spiral_region(
    available_cells,
    cluster_size,
    center
):
    center_x, center_y = center

    a = random.uniform(0.5, 1.5)
    b = random.uniform(1.2, 2.2)

    candidates = []

    theta = 0

    while len(candidates) < cluster_size * 4:

        r = a + b * theta

        x = int(
            round(center_x + r * math.cos(theta))
        )

        y = int(
            round(center_y + r * math.sin(theta))
        )

        if (
            clamp_grid(x, y) and
            (x, y) in available_cells
        ):

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:

                    nx = x + dx
                    ny = y + dy

                    if (
                        clamp_grid(nx, ny) and
                        (nx, ny) in available_cells
                    ):
                        candidates.append((nx, ny))

        theta += 0.18

        if theta > 8 * math.pi:
            break

    candidates = list(set(candidates))

    if len(candidates) < cluster_size:
        raise ValueError

    chosen = random.sample(
        candidates,
        cluster_size
    )

    for p in chosen:
        available_cells.remove(p)

    return chosen


# =========================================================
# INTERFACE DE SHAPE
# =========================================================
def generate_region(
    available_cells,
    cluster_size,
    shape,
    center
):
    for _ in range(30):

        try:

            if shape == "anel":
                return generate_ring_region(
                    available_cells,
                    cluster_size,
                    center
                )

            elif shape == "meia_lua":
                return generate_crescent_region(
                    available_cells,
                    cluster_size,
                    center
                )

            elif shape == "espiral":
                return generate_spiral_region(
                    available_cells,
                    cluster_size,
                    center
                )

        except ValueError:
            continue

    raise ValueError(
        f"Falha shape={shape}"
    )


# =========================================================
# GERAR REGIÕES ESPALHADAS + ENTRELAÇADAS
# =========================================================
available_cells = free_cells.copy()

regions_coords = []

zones = build_global_anchor_zones(
    GRID_SIZE,
    len(REGIONS)
)

print("===== REGIÕES =====")

i = 0

while i < len(REGIONS):

    cluster_size, shape, intervals = REGIONS[i]

    zone = zones[min(i, len(zones)-1)]

    primary_center = choose_center_in_zone(
        available_cells,
        zone
    )

    region = generate_region(
        available_cells,
        cluster_size,
        shape,
        primary_center
    )

    regions_coords.append(region)

    print(
        f"R{i+1}: shape={shape} centro={primary_center}"
    )

    # Tentativa de entrelaçamento
    if (
        i + 1 < len(REGIONS) and
        random.random() < 0.85
    ):

        next_cluster_size, next_shape, _ = REGIONS[i+1]

        try:

            secondary_center = choose_interlaced_center(
                primary_center,
                available_cells
            )

            region2 = generate_region(
                available_cells,
                next_cluster_size,
                next_shape,
                secondary_center
            )

            regions_coords.append(region2)

            print(
                f"R{i+2}: shape={next_shape} "
                f"centro={secondary_center} "
                f"entrelaçada"
            )

            i += 2
            continue

        except ValueError:
            pass

    i += 1


# =========================================================
# SALVAR COORDENADAS
# =========================================================
os.makedirs(
    os.path.dirname(ARQ_ENV_OUT),
    exist_ok=True
)

with open(
    ARQ_ENV_OUT,
    "w",
    encoding="utf-8"
) as f:

    for region in regions_coords:

        for x, y in sorted(region):
            f.write(f"{x},{y}\n")


# =========================================================
# REORGANIZAR VÍTIMAS
# =========================================================
remaining = df_input.copy()

ordered_blocks = []

print("\n===== VÍTIMAS =====")

for idx, (
    cluster_size,
    shape,
    intervals
) in enumerate(REGIONS):

    mask = False

    for low, high in intervals:

        current_mask = (
            (remaining["sobr"] >= low) &
            (remaining["sobr"] <= high)
        )

        mask = mask | current_mask

    candidatos = remaining[mask].copy()

    if len(candidatos) >= cluster_size:

        bloco = candidatos.sort_values(
            by="sobr",
            ascending=False
        ).head(cluster_size)

    else:

        faltam = cluster_size - len(candidatos)

        usados = set(candidatos.index)

        restantes = remaining[
            ~remaining.index.isin(usados)
        ].copy()

        def min_distance(v):

            dists = []

            for low, high in intervals:

                if low <= v <= high:
                    dists.append(0)

                else:
                    centro = (low + high) / 2
                    dists.append(abs(v - centro))

            return min(dists)

        complemento = restantes.iloc[
                restantes["sobr"].apply(min_distance).argsort()
        ].head(faltam)

        bloco = pd.concat(
            [candidatos, complemento]
        )

    remaining = remaining[
        ~remaining.index.isin(bloco.index)
    ]

    ordered_blocks.append(bloco)
    
    

    print(
        f"R{idx+1}: {len(bloco)} vítimas"
    )

    # =====================================================
    # ESTATÍSTICAS DA REGIÃO DENSA
    # =====================================================
    tri_counts = bloco["tri"].value_counts().sort_index()

    print(f"\n--- REGIÃO R{idx+1} ({shape}) ---")

    for tri_value, tri_count in tri_counts.items():
        print(
            f"tri={tri_value}: {tri_count} vítimas"
        )

    print(
        f"sobr média: {bloco['sobr'].mean():.4f}"
    )

    print(
        f"sobr desvio padrão: "
        f"{bloco['sobr'].std():.4f}"
    )
    
# =========================================================
# SOBRAS
# =========================================================
if not remaining.empty:
    ordered_blocks.append(remaining)


# =========================================================
# SALVAR DATA
# =========================================================
df_final = pd.concat(
    ordered_blocks,
    ignore_index=True
)

os.makedirs(
    os.path.dirname(ARQ_DATA_OUT),
    exist_ok=True
)

df_final.to_csv(
    ARQ_DATA_OUT,
    index=False
)


# =========================================================
# FINAL
# =========================================================
print("\n===== FINAL =====")
print(
    f"env_victims: {len(df_final)}"
)
print(
    f"data.csv: {len(df_final)}"
)