import pandas as pd
import numpy as np
import plotly.graph_objects as go
from google.colab import files


# ============================================================
# 1. CARGA Y CLASIFICACIÓN DE ARCHIVOS
# ============================================================
def cargar_y_clasificar_archivos():
    global df_checkin, df_ventas, df_visitas

    CHECK_IN_HEADERS = [
        "Nombre del Rep. Ventas", "Visitas planificadas", "Visitas completadas", "Primer check-in",
        "Promedio primer check-in MTD", "Último check-out", "Promedio último check-out MTD",
        "Duración promedio de la visita (min:sec)", "Duración promedio de la visita MTD (min:sec)",
        "Tiempo total dentro de los PDV (H)", "Tiempo total dentro de los PDV MTD (H)",
        "Ruta Efectiva", "Ruta Efectiva MTD"
    ]

    VENTAS_HEADERS = [
        "bdr_id", "Orders", "Total Revenue", "# Orders BEES Force", "# Orders BEES Link",
        "# Orders BEES Customer", "# Orders BEES Grow", "Revenue BEES Force", "Revenue BEES Link",
        "Revenue BEES Customer", "Revenue BEES Grow"
    ]

    VISITAS_HEADERS = [
        "Nombre del Rep. Ventas", "Visitas planificadas", "Visitas completadas", "Visita con justificacion",
        "GPS Ok visitas", "% GPS Ok visitas", "% GPS Ok visitas MTD", "GPS Ok > 2 min Visitas",
        "% GPS Ok > 2 min visitas", "% GPS Ok > 2 min visitas MTD", "GPS > 2 min + Justificadas GPS Ok",
        "% GPS > 2 min + Justificadas GPS Ok", "% GPS > 2 min + Justificadas GPS Ok MTD",
        "Visitas planificadas con pedidos", "GPS OK con pedidos"
    ]

    print("Sube los tres archivos (Check_In, Ventas y Visitas)...")
    uploaded = files.upload()

    if len(uploaded) != 3:
        raise ValueError(f"Se esperaban 3 archivos, pero se recibieron {len(uploaded)}")

    df_checkin = df_ventas = df_visitas = None

    for name in uploaded.keys():
        ext = name.split('.')[-1].lower()
        df = pd.read_excel(name) if ext in ["xlsx", "xls"] else pd.read_csv(name) if ext == "csv" else None
        if df is None:
            print(f"{name}: formato no soportado, omitido")
            continue

        headers = list(df.columns)

        if headers == CHECK_IN_HEADERS:
            df_checkin = df
            print(f"{name}: identificado como Check_In")
        elif headers == VENTAS_HEADERS:
            df_ventas = df
            print(f"{name}: identificado como Ventas")
        elif headers == VISITAS_HEADERS:
            df_visitas = df
            print(f"{name}: identificado como Visitas")
        else:
            print(f"{name}: encabezados no coinciden con ningún tipo conocido")

    if any(x is None for x in [df_checkin, df_ventas, df_visitas]):
        raise ValueError("Falta al menos uno de los archivos requeridos (Check_In, Ventas, Visitas).")

    print("✓ Archivos identificados correctamente y guardados en memoria.")
    return df_checkin, df_ventas, df_visitas


# ============================================================
# 2. LIMPIEZA BÁSICA
# ============================================================
def limpiar_df(df):
    if df.shape[1] >= 2:
        df = df.dropna(subset=[df.columns[1]])
    return df


# ============================================================
# 3. NORMALIZAR NOMBRE Y CÓDIGO
# ============================================================
def separar_nombre_codigo(df):
    if "Nombre del Rep. Ventas" in df.columns:
        df[["Rep. Ventas", "Codigo"]] = df["Nombre del Rep. Ventas"].str.split(" - ", n=1, expand=True)
        df["Codigo"] = df["Codigo"].fillna(df["Rep. Ventas"])
        df.drop(columns=["Nombre del Rep. Ventas"], inplace=True)
    return df


# ============================================================
# 4. MERGE Y FILTRADO
# ============================================================
def unir_tablas(df_checkin, df_ventas, df_visitas):
    df_merge = pd.merge(
        df_checkin[
            ["Codigo", "Rep. Ventas", "Visitas planificadas", "Visitas completadas", "Primer check-in"]
        ],
        df_visitas[
            ["Codigo", "GPS Ok visitas", "% GPS Ok visitas", "GPS Ok > 2 min Visitas", "% GPS Ok > 2 min visitas"]
        ],
        on="Codigo", how="outer"
    )

    df_merge = pd.merge(
        df_merge,
        df_ventas[["bdr_id", "Orders", "Total Revenue"]],
        left_on="Codigo", right_on="bdr_id", how="left"
    )

    df_merge = df_merge[
        [
            "Rep. Ventas", "Codigo", "Orders", "Total Revenue",
            "Visitas planificadas", "Visitas completadas",
            "GPS Ok visitas", "% GPS Ok visitas",
            "GPS Ok > 2 min Visitas", "% GPS Ok > 2 min visitas",
            "Primer check-in"
        ]
    ]
    return df_merge


# ============================================================
# 5. FILTRAR CÓDIGOS Y LIMPIAR VALORES
# ============================================================
def filtrar_codigos(df, codigos):
    df = df[df["Codigo"].isin(codigos)].drop(columns=["Codigo"])
    df["Primer check-in"] = df["Primer check-in"].fillna("-")
    df.fillna(0, inplace=True)
    return df


# ============================================================
# 6. TABLA CON COLORES (VISUAL)
# ============================================================
def crear_tabla_indicadores(df, width=1000, height=550):
    def gradient_color(values):
        vals = np.array(values, dtype=float)
        min_v, max_v, mean_v = np.nanmin(vals), np.nanmax(vals), np.nanmean(vals)
        colors = []
        for v in vals:
            if np.isnan(v):
                colors.append('white')
                continue
            if v <= mean_v:
                ratio = (v - min_v) / (mean_v - min_v) if mean_v != min_v else 0
                r = int(248 + (251 - 248) * ratio)
                g = int(105 + (233 - 105) * ratio)
                b = int(108 + (130 - 108) * ratio)
            else:
                ratio = (v - mean_v) / (max_v - mean_v) if max_v != mean_v else 0
                r = int(251 + (99 - 251) * ratio)
                g = int(233 + (190 - 233) * ratio)
                b = int(130 + (123 - 130) * ratio)
            colors.append(f'rgb({r},{g},{b})')
        return colors
        
    def parse_percent(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            return float(x.strip('%'))
        return float(x)
        
    gps_ok_colors = gradient_color([parse_percent(x) for x in df["% GPS Ok visitas"]])
    gps_ok2_colors = gradient_color([parse_percent(x) for x in df["% GPS Ok > 2 min visitas"]])

    times = []
    for t in df["Primer check-in"]:
        if not isinstance(t, str) or ':' not in t:
            times.append(None)
            continue
        parts = t.strip().split()
        hora = parts[0]
        am_pm = parts[1].lower() if len(parts) > 1 else ''
        try:
            h, m, s = map(int, hora.split(':'))
        except ValueError:
            times.append(None)
            continue
        if 'p.m.' in am_pm and h != 12:
            h += 12
        if 'a.m.' in am_pm and h == 12:
            h = 0
        times.append(h * 3600 + m * 60 + s)
    min_t, max_t = min(filter(None, times)), max(filter(None, times))
    checkin_colors = []
    for t in times:
        if t is None:
            checkin_colors.append('white')
        elif t == min_t:
            checkin_colors.append('rgb(99,190,123)')
        elif t == max_t:
            checkin_colors.append('rgb(248,105,108)')
        else:
            checkin_colors.append('white')

    header_colors = ['lightgray'] * len(df.columns)
    col_index = df.columns.get_loc("% GPS Ok > 2 min visitas")
    header_colors[col_index] = '#A1D1FE'

    fill_colors = [
        ['white'] * len(df),
        ['white'] * len(df),
        ['white'] * len(df),
        ['white'] * len(df),
        ['white'] * len(df),
        gps_ok_colors,
        ['white'] * len(df),
        gps_ok2_colors,
        checkin_colors
    ]

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                    fill_color=header_colors,
                    align='left',
                    font=dict(color='black', size=12)),
        cells=dict(values=[df[c] for c in df.columns],
                   fill_color=fill_colors,
                   align='left',
                   font=dict(color='black', size=11))
    )])
    fig.update_layout(width=width, height=height)

    return fig
# ============================================================
# 7. PIPELINE                     
# ============================================================
def ejecutar_pipeline(df_checkin, df_ventas, df_visitas, codigos, width=1000, height=550):

    df_checkin = separar_nombre_codigo(limpiar_df(df_checkin))
    df_ventas = separar_nombre_codigo(limpiar_df(df_ventas))
    df_visitas = separar_nombre_codigo(limpiar_df(df_visitas))

    df_merge = unir_tablas(df_checkin, df_ventas, df_visitas)
    df_filtrado = filtrar_codigos(df_merge, codigos)

    fig = crear_tabla_indicadores(df_filtrado, width=width, height=height)
    print("Pipeline completado.")
    return df_filtrado, fig


