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

    CHECK_IN_MTD = [
        "Nombre del Rep. Ventas", "Fecha", "Visitas planificadas", "Visitas completadas", "Primer check-in",
        "Último check-out", "Duración promedio de la visita (min:sec)", "Tiempo total dentro de los PDV (H)", "Ruta Efectiva"
    ]

    VISITAS_MTD = [
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
        if temp.shape[1] == 1:
            temp[1] = None
        
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

    df_users = df_checkin[["Codigo", "Rep. Ventas"]].drop_duplicates().copy()
    
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

    return df_merge, df_users

# ============================================================
# 5. FILTRAR CÓDIGOS, LIMPIAR VALORES Y SUBTOTAL
# ============================================================
def filtrar_codigos(df, codigos):
    df = df[df["Codigo"].isin(codigos)].drop(columns=["Codigo"])
    df["Primer check-in"] = df["Primer check-in"].fillna("-")

    subtotal = { 
        "Equipo": "-",
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

def gradient_color(values, thresholds=(30, 60, 80)):
    t1, t2, t3 = thresholds
    colors = []

    for v in values:
        if np.isnan(v):
            colors.append("white")
            continue

        if v < t1:
            colors.append("rgb(248,105,108)")
        elif v < t2:
            colors.append("rgb(251,233,130)")
        elif v < t3:
            colors.append("rgb(251,190,123)")
        else:
            colors.append("rgb(99,190,123)")
    return colors

def generar_fill_colors(df, colores_especiales):
    fill_colors = []
    for col in df.columns:
        if col in colores_especiales:
             fill_colors.append(colores_especiales[col])
        else:
             fill_colors.append(['white'] * len(df))
    return fill_colors
    
def crear_tabla_indicadores(df, venta=1, width=1000, height=550):
    if venta == 0:
        cols = [
            "Equipo", "Rep. Ventas",
            "Visitas planificadas",
            "GPS Ok visitas", "% GPS Ok visitas",
            "GPS Ok > 2 min Visitas", "% GPS Ok > 2 min visitas",
            "Primer check-in"
        ]
    else:
        cols = [
            "Equipo", "Rep. Ventas", "Orders", 
            "Total Revenue", "Visitas planificadas",
            "GPS Ok visitas", "% GPS Ok visitas",
            "GPS Ok > 2 min Visitas", "% GPS Ok > 2 min visitas",
            "Primer check-in"
        ]

    df = df[cols]
    
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

def agregar_equipo(df, df_usuarios, mapa_equipo, adopcion=0):
    df = df.copy()
    df_usuarios = df_usuarios.copy()
    
    df_usuarios["Equipo"] = df_usuarios["Codigo"].map(mapa_equipo)
    
    match int(adopcion):
        case 0:
            df = df.merge(
                df_usuarios[["Rep. Ventas", "Equipo"]],
                on="Rep. Ventas",
                how="left"
            )

        case 1:
            df = df.merge(
                df_usuarios,
                left_on="CodVendedor",
                right_on="Codigo",
                how="left"
            )

        case 2:
            df = df.merge(
                df_usuarios,
                left_on="BDR ID",
                right_on="Codigo",
                how="left"
            ).drop(columns="Codigo")

        case _:
            raise ValueError("adopcion debe ser 0, 1 o 2")

    
    cols = ["Equipo"] + [c for c in df.columns if c != "Equipo"]
    df = df[cols]
    return df
# ============================================================
# 7. PIPELINE                     
# ============================================================
def ejecutar_pipeline(df_checkin, df_visitas, df_ventas, codigos, mapa_equipo, venta = 1, width=1000, height=550):

    df_checkin = separar_nombre_codigo(limpiar_df(df_checkin))
    df_visitas = separar_nombre_codigo(limpiar_df(df_visitas))
    df_ventas = separar_nombre_codigo(limpiar_df(df_ventas))
    
    df_merge, df_users = unir_tablas(df_checkin, df_visitas, df_ventas, venta=venta)
    df_merge_user = agregar_equipo(df_merge, df_users, mapa_equipo)
    
    for c in codigos:
      df_filtrado = filtrar_codigos(df_merge_user, c)
      fig = crear_tabla_indicadores(df_filtrado, venta=venta, width=width, height=height)
      fig.show()
        
    print("Pipeline completado.")
    return df_filtrado, df_users

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

# ============================================================
# 9. PROCESAMIENTO VENDEDOR                     
# ============================================================

def cargar_archivos():
  columnas_str = {
    "codcia":str,
    "codclte":str,
    "domic":str,
    "numero_pedido":str,
    "codigoproducto":str,
    "vendedor":str,
    "Cod Cliente":str,
    "Clave del Cliente (15 dígitos)":str
  }
  uploaded = files.upload()
  all_dfs = []
  for name in uploaded.keys():
        ext = name.split('.')[-1].lower()
        df = pd.read_excel(name, dtype={c: str for c in columnas_str}) if ext in ["xlsx", "xls"] else pd.read_csv(name) if ext == "csv" else None
        if df is None:
            print(f"{name}: formato no soportado, omitido")
            continue
        all_dfs.append(df)

  if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df
  else:
    print("No se cargaron archivos.")
    return None

def agregar_ceros(df):
  columnas_longitud = {'codcia': 3, 'domic': 3,'codclte': 6, 'vendedor': 5}
  for columna, longitud in columnas_longitud.items():
    if columna in df.columns:
      df[columna] = df[columna].astype(str).str.zfill(longitud)
  return df

def procesar_df(df, codigos_permitidos=None):
  df = df.copy()
  encabezado = [
      "fecha",
      "Cod Cliente",
      "nombrecliente",
      "vendedor",
      "nombrevendedor",
      "vtadvo",
      "canal",
      "tipo_pedido",
      "numero_pedido",
      "DescripcionCategoria",
      "codigoproducto",
      "descrpcionproducto",
      "cajas",
      "soles"
    ]
  if 'Cod Cliente' not in df.columns:
    df = agregar_ceros(df)
    df['Cod Cliente'] = df['codcia'] + '001' + df['codclte'] + df['domic']

  df = df[encabezado]
  df['tipo_pedido'] = df['tipo_pedido'].apply(lambda x: 'NON BEES' if not isinstance(x, str) or x[:3] != 'B2B' else x)

  mask_producto = (
        ~df["codigoproducto"].astype(str).str.startswith("70") &
        ~df["codigoproducto"].astype(str).str.startswith("90")
  )
  if codigos_permitidos is not None:
        mask_producto |= df["codigoproducto"].astype(str).isin(codigos_permitidos)

  df = df[
      (df["tipo_pedido"] != "B2B_WEB") &
      (df["vtadvo"] == "V") &
      (df["canal"] == "BODEGA") &
      mask_producto
  ]
  df["fecha"] = pd.to_datetime(df["fecha"], format="%Y/%m/%d", errors="coerce")
  df["mes"] = df["fecha"].dt.to_period("M").astype(str)
  df["mes_n"] = df['mes'].astype(str).str[-2:]
  df['DT'] = df["Cod Cliente"].astype(str).str[:3]
  df['CodVendedor'] = df['DT'] + '001-' + df['vendedor']
  return df

def modelado(df, df_users):
  df_modelado = df.copy()

  df_modelado = df_modelado[df_modelado['CodVendedor'].isin(df_users['Codigo'])]
  df_modelado_tipo = df_modelado.groupby(["CodVendedor", "tipo_pedido"]).agg(
            pedidos=("numero_pedido", "nunique")
        ).reset_index()

  df_modelado_tipo_p = df_modelado_tipo.pivot_table(
      index= ["CodVendedor"],
      columns= ["tipo_pedido"],
      values= ["pedidos"],
      aggfunc= "sum"
  ).reset_index()

  df_modelado_tipo_p.fillna(0, inplace=True)

  df_modelado_tipo_p.columns = [
      f"{col1}_{col2}" if col2 != "" else col1
      for col1, col2 in df_modelado_tipo_p.columns.to_flat_index()
  ]
  df_modelado_tipo_p['pedidos_total'] = 0

  for c in df_modelado_tipo_p.columns:
    map = c.split("_")
    if  len(map) > 1 and map[0] == 'pedidos' and map[1] != 'total':
      df_modelado_tipo_p['pedidos_total'] += df_modelado_tipo_p[c]

  df_modelado_tipo_p['Adopción BEES'] = (df_modelado_tipo_p['pedidos_B2B_APP'] + df_modelado_tipo_p['pedidos_B2B_FORCE'])/df_modelado_tipo_p['pedidos_total']
  df_modelado_tipo_p['Adopción NON BEES'] = 1 - df_modelado_tipo_p['Adopción BEES']

  df_modelado_tipo_p["Adopción BEES"] = df_modelado_tipo_p["Adopción BEES"].apply(
        lambda x: f"{x*100:.2f}%" if x <= 1 else f"{x:.2f}%"
    )
  df_modelado_tipo_p["Adopción NON BEES"] = df_modelado_tipo_p["Adopción NON BEES"].apply(
        lambda x: f"{x*100:.2f}%" if x <= 1 else f"{x:.2f}%"
  )
  return df_modelado_tipo_p

def crear_tabla_adopcion(df, width=1000, height=550):
    cols = [
        "Equipo",
        "Rep. Ventas",
        "pedidos_B2B_APP",
        "pedidos_B2B_FORCE",
        "pedidos_NON BEES",
        "Adopción BEES"
    ]

    df = df[cols]
    
    df = df.assign(
      temp_val=pd.to_numeric(df["Adopción BEES"].astype(str).str.replace("%", ""), errors="coerce")
    ).sort_values("temp_val", ascending=False).drop(columns="temp_val").reset_index(drop=True)
    
    adop_ok_colors = gradient_color([float(x.strip('%')) for x in df["Adopción BEES"]])
    #adop_ok2_colors = gradient_color([float(x.strip('%')) for x in df["Adopción NON BEES"]])

    header_colors = ['lightgray'] * len(df.columns)
    col_index = df.columns.get_loc("Adopción BEES")
    header_colors[col_index] = '#A1D1FE'

    colores_especiales = {
        "Adopción BEES": adop_ok_colors
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

def AdopcionVendedores(df_users, mapa_equipo, width=1000, height=550):
    df = cargar_archivos()
    df_procesado = agregar_equipo(modelado(procesar_df(agregar_ceros(df)),df_users), df_users, mapa_equipo, adopcion=1)
    df_procesado = df_procesado[df_procesado["Equipo"].notna()]
    
    fig = crear_tabla_adopcion(df_procesado, width=width, height=height)
    fig.show()
    return df_procesado

# ============================================================
# 10. TAREAS                                    
# ============================================================

def formato_tareas(df, df_users, mapa_equipo,):
    def separar_csv(df_tareas):
        df_tareas_2 = df_tareas.copy()
        df_tareas_2 = df_tareas.iloc[:, 0].str.split(";", expand=True)
        col = df_tareas.columns[0].split(";")
        df_tareas_2.columns= col
        return df_tareas_2

    def gradient_colors(values, thresholds=(0.30, 0.60, 0.80)):
        t1, t2, t3 = thresholds
        styles = []

        for v in values:
            if pd.isna(v):
                styles.append("")
                continue

            if v < t1:
                styles.append("background-color: rgb(248,105,108)")
            elif v < t2:
                styles.append("background-color: rgb(251,233,130)")
            elif v < t3:
                styles.append("background-color: rgb(251,190,123)")
            else:
                styles.append("background-color: rgb(99,190,123)")
        return styles

    tarea = agregar_equipo(separar_csv(df.copy()), df_users.copy(), mapa_equipo.copy(), adopcion=2)
    tarea = tarea[tarea["Equipo"].notna()]
    
    out = (
        tarea
        .groupby(["Equipo", "Rep. Ventas", "Task Name"], as_index=False)
        .agg(
            total=("Is Task Effective", "count"),
            true_count=("Is Task Effective", lambda x: (x == 'true').sum())
        )
    )
    
    out["%Alcance"] = out["true_count"] / out["total"]
    out = out.rename(columns={"true_count": "tarea efectiva"})
    
    result = out.drop(columns="total").melt(
        id_vars=["Equipo", "Rep. Ventas", "Task Name"],
        value_vars=["tarea efectiva", "%Alcance"],
    ).pivot_table(
        index=["Equipo", "Rep. Ventas"],
        columns=["Task Name", "variable"],
        values=["value"],
        aggfunc= "sum"
    ).fillna(0)

    result.columns = result.columns.droplevel(0)

    styled = (
        result.head(10)
        .style
        .format(
            "{:.2%}",
            subset=pd.IndexSlice[:, pd.IndexSlice[:, "%Alcance"]]
        )
        .format(
            "{:.0f}",
            subset=pd.IndexSlice[:, pd.IndexSlice[:, "tarea efectiva"]]
        )
        
        .apply(
            gradient_colors,
            subset=pd.IndexSlice[:, pd.IndexSlice[:, "%Alcance"]],
            axis=0
        )
        .set_properties(**{
            'font-family': 'Arial, sans-serif',
            'font-size': '12px',
            'border': '2px solid white', 
            'text-align': 'center',
            'vertical-align': 'middle'
        })
        .set_table_styles([
            {
                'selector': 'th, td',
                'props': [
                    ('font-family', 'Arial, sans-serif'),
                    ('border', '1px solid white !important'),
                    ('text-align', 'center'),
                    ('padding', '8px')
                ]
            },
            {
                "selector": "th.col_heading",
                "props": [
                    ("background-color", "#e0e0e0"),
                    ("color", "black"),
                    ("font-weight", "bold"),
                    ("border", "1px solid #bdbdbd")
                ]
            },
            {
                "selector": "th.row_heading",
                "props": [
                    ("background-color", "#e0e0e0"),
                    ("color", "black"),
                    ("font-weight", "bold"),
                    ("border", "1px solid #282A2C")
                ]
            },
            {
                "selector": "th.blank",
                "props": [
                    ("background-color", "#e0e0e0"),
                    ("border", "1px solid #282A2C")
                ]
            },
            {
                'selector': 'th',
                'props': [
                    ('background-color', '#d9d9d9'),
                    ('color', 'black'),
                    ('font-family', 'Arial, sans-serif'),
                    ('font-weight', 'bold'),
                    ('border', '1px solid #7f7f7f'),
                    ('text-align', 'center')
                ]
            },
            {
                'selector': 'td',
                'props': [
                    ('border', '1px solid white'),
                    ('font-family', 'Arial, sans-serif'),
                    ('text-align', 'center'),
                    ('font-weight', 'normal')
                ]
            },
            {
                'selector': '',
                'props': [
                    ('border-collapse', 'collapse'),
                    ('border', 'none')
                ]
            }
        ])
    )

    return styled



















