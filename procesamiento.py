import pandas as pd
import numpy as np
import plotly.graph_objects as go
from google.colab import files
from datetime import datetime

def cargar_y_clasificar_archivos():

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

    VISITAS_MTD = [
        "Nombre del Rep. Ventas", "Fecha", "Visitas planificadas", "Visitas completadas", "Primer check-in",
        "Último check-out", "Duración promedio de la visita (min:sec)", "Tiempo total dentro de los PDV (H)", "Ruta Efectiva"
    ]

    CHECK_IN_MTD = [
        "Rep. Ventas", "Fecha", "Visitas planificadas", "Visitas completadas", "GPS Ok visitas", "% GPS Ok visitas",
        "GPS Ok > 2 min Visitas", "% GPS Ok > 2 min visitas", "GPS > 2 min + Justificadas GPS Ok", "% GPS > 2 min + Justificadas GPS Ok", 
        "Visita con justificacion", "Visitas planificadas con pedidos", "GPS OK con pedidos"
    ]

    print("Sube los tres archivos (Check_In, Ventas y Visitas)...")
    uploaded = files.upload()

    if not (2 <= len(uploaded) <= 3):
        raise ValueError(f"Se esperaban 2-3 archivos, pero se recibieron {len(uploaded)}")

    df_checkin = df_ventas = df_visitas = None

    for name in uploaded.keys():
        ext = name.split('.')[-1].lower()
        df = pd.read_excel(name) if ext in ["xlsx", "xls"] else pd.read_csv(name) if ext == "csv" else None
        if df is None:
            print(f"{name}: formato no soportado, omitido")
            continue

        headers = list(df.columns)

        if headers == CHECK_IN_HEADERS or headers == CHECK_IN_MTD:
            df_checkin = df
            print(f"{name}: identificado como Check_In")
        elif headers == VENTAS_HEADERS:
            df_ventas = df
            print(f"{name}: identificado como Ventas")
        elif headers == VISITAS_HEADERS or headers == VISITAS_MTD:
            df_visitas = df
            print(f"{name}: identificado como Visitas")
        else:
            print(f"{name}: encabezados no coinciden con ningún tipo conocido")

    if any(x is None for x in [df_checkin, df_visitas]):
        raise ValueError("Falta al menos uno de los archivos requeridos (Check_In, Visitas).")

    print("✓ Archivos identificados correctamente y guardados en memoria.")
    return df_checkin, df_ventas, df_visitas


# ============================================================
# 2. LIMPIEZA BÁSICA
# ============================================================
def limpiar_df(df):
    if df is None:
        return None
    if df.shape[1] >= 2:
        df = df.dropna(subset=[df.columns[1]])
    return df

# ============================================================
# 3. NORMALIZAR NOMBRE Y CÓDIGO
# ============================================================
def separar_nombre_codigo(df):
    if df is None:
      return None
    col = None
    if "Nombre del Rep. Ventas" in df.columns:
        col = "Nombre del Rep. Ventas"
    elif "Rep. Ventas" in df.columns:
        col = "Rep. Ventas"

    if col:
        temp = df[col].str.split(" - ", n=1, expand=True)
        rep = temp[0]
        codigo = temp[1].fillna(rep)
        df["Rep. Ventas"] = rep
        df["Codigo"] = codigo
        if col != "Rep. Ventas":
            df.drop(columns=[col], inplace=True)

    return df

# ============================================================
# 4. MERGE Y FILTRADO
# ============================================================
def unir_tablas(df_checkin, df_visitas, df_ventas=None, venta=1):
    if df_checkin is None or df_visitas is None:
        raise ValueError("df_checkin y df_visitas no pueden ser None")
    df_merge = pd.merge(
        df_checkin[
            ["Codigo", "Rep. Ventas", "Visitas planificadas", "Visitas completadas", "Primer check-in"]
        ],
        df_visitas[
            ["Codigo", "GPS Ok visitas", "% GPS Ok visitas", "GPS Ok > 2 min Visitas", "% GPS Ok > 2 min visitas"]
        ],
        on="Codigo", how="outer"
    )

    if venta == 1:
        df_merge = pd.merge(
            df_merge,
            df_ventas[["bdr_id", "Orders", "Total Revenue"]],
            left_on="Codigo",
            right_on="bdr_id",
            how="left"
        )
        df_merge.drop(columns=["bdr_id"], inplace=True)
    else:
        df_merge["Orders"] = 0
        df_merge["Total Revenue"] = 0

    df_merge = df_merge[
        [
            "Rep. Ventas", "Codigo", "Orders", "Total Revenue",
            "Visitas planificadas", "Visitas completadas",
            "GPS Ok visitas", "% GPS Ok visitas",
            "GPS Ok > 2 min Visitas", "% GPS Ok > 2 min visitas",
            "Primer check-in"
        ]
    ]

    df_merge["Rep. Ventas"] = df_merge["Rep. Ventas"].str.upper()
    
    df_merge["% GPS Ok visitas"] = df_merge["% GPS Ok visitas"].apply(
        lambda x: f"{x*100:.2f}%" if x <= 1 else f"{x:.2f}%"
    )
    df_merge["% GPS Ok > 2 min visitas"] = df_merge["% GPS Ok > 2 min visitas"].apply(
        lambda x: f"{x*100:.2f}%" if x <= 1 else f"{x:.2f}%"
    )
    for col in ["Orders", "Total Revenue", "Visitas planificadas", "Visitas completadas", "GPS Ok visitas", "GPS Ok > 2 min Visitas"]:
      df_merge[col] = pd.to_numeric(df_merge[col], errors="coerce").fillna(0)

    return df_merge

# ============================================================
# 5. FILTRAR CÓDIGOS, LIMPIAR VALORES Y SUBTOTAL
# ============================================================
def filtrar_codigos(df, codigos):
    df = df[df["Codigo"].isin(codigos)].drop(columns=["Codigo"])
    df["Primer check-in"] = df["Primer check-in"].fillna("-")

    subtotal = { 
        "Rep. Ventas": "TOTAL", 
        "Orders": df["Orders"].sum(), 
        "Total Revenue": df["Total Revenue"].sum(),
        "Visitas planificadas": df["Visitas planificadas"].sum(), 
        "Visitas completadas": df["Visitas completadas"].sum(), 
        "GPS Ok visitas": df["GPS Ok visitas"].sum(), 
        "% GPS Ok visitas": f"{df['GPS Ok visitas'].sum() / df['Visitas planificadas'].sum() * 100:.2f}%", 
        "GPS Ok > 2 min Visitas": df["GPS Ok > 2 min Visitas"].sum(), 
        "% GPS Ok > 2 min visitas": f"{df['GPS Ok > 2 min Visitas'].sum() / df['Visitas planificadas'].sum() * 100:.2f}%", 
        "Primer check-in": "-" 
        } 
    
    df = df.assign(
      temp_val=pd.to_numeric(df["% GPS Ok > 2 min visitas"].astype(str).str.replace("%", ""), errors="coerce")
    ).sort_values("temp_val", ascending=False).drop(columns="temp_val").reset_index(drop=True)
    df["Rep. Ventas"] = df["Rep. Ventas"].str.upper()
    df = pd.concat([df, pd.DataFrame([subtotal])], ignore_index=True)
    df["Total Revenue"] = df["Total Revenue"].apply(lambda x: f"{x:,.2f}")

    return df

# ============================================================
# 6. TABLA CON COLORES (VISUAL)
# ============================================================
def crear_tabla_indicadores(df, venta=1, width=1000, height=550):
    if venta == 0:
        cols = [
            "Rep. Ventas",
            "Visitas planificadas", "Visitas completadas",
            "GPS Ok visitas", "% GPS Ok visitas",
            "GPS Ok > 2 min Visitas", "% GPS Ok > 2 min visitas",
            "Primer check-in"
        ]
    else:
        cols = [
            "Rep. Ventas", "Orders", "Total Revenue",
            "Visitas planificadas", "Visitas completadas",
            "GPS Ok visitas", "% GPS Ok visitas",
            "GPS Ok > 2 min Visitas", "% GPS Ok > 2 min visitas",
            "Primer check-in"
        ]

    df = df[cols]
  
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
        
    gps_ok_colors = gradient_color([float(x.strip('%')) for x in df["% GPS Ok visitas"]])
    gps_ok2_colors = gradient_color([float(x.strip('%')) for x in df["% GPS Ok > 2 min visitas"]])

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

    valid_times = [t for t in times if t is not None]
    if valid_times:
      min_t, max_t = min(valid_times), max(valid_times)
    else:
      min_t, max_t = None, None
    #min_t, max_t = min(filter(None, times)), max(filter(None, times))
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

    def generar_fill_colors(df, colores_especiales):
        fill_colors = []
        for col in df.columns:
            if col in colores_especiales:
                fill_colors.append(colores_especiales[col])
            else:
                fill_colors.append(['white'] * len(df))
        return fill_colors

    colores_especiales = {
        "% GPS Ok visitas": gps_ok_colors,
        "% GPS Ok > 2 min visitas": gps_ok2_colors,
        "Primer check-in": checkin_colors
    }

    fill_colors = generar_fill_colors(df, colores_especiales)

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
def ejecutar_pipeline(df_checkin, df_visitas, df_ventas, codigos, venta = 1, width=1000, height=550):

    df_checkin = separar_nombre_codigo(limpiar_df(df_checkin))
    df_visitas = separar_nombre_codigo(limpiar_df(df_visitas))
    df_ventas = separar_nombre_codigo(limpiar_df(df_ventas))
    
    df_merge = unir_tablas(df_checkin, df_visitas, df_ventas, venta=venta)

    df_filtrado = filtrar_codigos(df_merge, codigos)

    fig = crear_tabla_indicadores(df_filtrado, venta=venta, width=width, height=height)
    print("Pipeline completado.")
    return df_filtrado, fig

# ============================================================
# 8. DESCARGAR EXCEL                     
# ============================================================
def generar_excel(df_xl, nombre_archivo = None):
    """
    Guarda el DataFrame procesado como Excel y devuelve el nombre de archivo generado.
    Si no se pasa nombre_archivo, se genera uno con formato 'mes ddmmyyyy.xlsx'.
    """
    if not nombre_archivo:
        fecha_actual = datetime.now()
        nombre_archivo = f"resultado-{fecha_actual.strftime('%d%m%Y')}.xlsx"
    if not nombre_archivo.lower().endswith(".xlsx"):
        nombre_archivo += ".xlsx"
    df_xl.to_excel(nombre_archivo, index=False, engine="openpyxl")

    return nombre_archivo



