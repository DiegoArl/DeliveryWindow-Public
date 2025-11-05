import pandas as pd

def construir_df(ruta_excel):
    """
    Lee el archivo Excel original y construye el DataFrame final df_csv.
    Parámetro:
        ruta_excel: str -> ruta o nombre del archivo Excel subido
    Retorna:
        DataFrame df_csv con estructura final lista para exportar a CSV.
    """
    # Leer datos
    df = pd.read_excel(ruta_excel)

    # --- Limpieza y normalización de los días ---
    df_filtrado = df[["COD CLIENTE BEES", "dia_visita"]].copy()
    df_filtrado["dia_visita"] = (
        df_filtrado["dia_visita"]
        .astype(str)
        .str.replace(" ", "", regex=False)
        .str.upper()
    )

    dias_semana = ["LUNES", "MARTES", "MIERCOLES", "JUEVES", "VIERNES", "SABADO", "DOMINGO"]

    df_filtrado["dia_visita"] = df_filtrado["dia_visita"].apply(
        lambda x: dias_semana if "TODOSLOSDIAS" in str(x) else str(x)
    )
    df_filtrado["dia_visita"] = df_filtrado["dia_visita"].apply(
        lambda x: x if isinstance(x, list) else str(x).split("Y")
    )
    df_filtrado = df_filtrado.explode("dia_visita", ignore_index=True)
    df_filtrado["dia_visita"] = df_filtrado["dia_visita"].str.strip().str.upper()
    df_filtrado = df_filtrado.drop_duplicates().reset_index(drop=True)

    # --- Base para combinar ---
    df_base = df[["codigo_cliente", "COD CLIENTE BEES"]].drop_duplicates().reset_index(drop=True)

    # --- Estructura final ---
    columnas = [
        "UNB", "ClientCode", "Document", "Exception",
        "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun",
        "MinValueMon", "MinValueTue", "MinValueWed", "MinValueThu", "MinValueFri", "MinValueSat", "MinValueSun",
        "AddAmountMon", "AddAmountTue", "AddAmountWed", "AddAmountThu", "AddAmountFri", "AddAmountSat", "AddAmountSun",
        "DeliveryFrequency", "ClickAndCollectActive"
    ]

    df_csv = pd.DataFrame(columns=columnas)
    df_csv["UNB"] = df_base["COD CLIENTE BEES"].str[2:10]
    df_csv["ClientCode"] = df_base["codigo_cliente"]
    df_csv["Document"] = df_base["COD CLIENTE BEES"]
    df_csv["Exception"] = True
    df_csv[["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]] = 0
    df_csv[[
        "MinValueMon","MinValueTue","MinValueWed","MinValueThu","MinValueFri","MinValueSat","MinValueSun",
        "AddAmountMon","AddAmountTue","AddAmountWed","AddAmountThu","AddAmountFri","AddAmountSat","AddAmountSun"
    ]] = 0
    df_csv["DeliveryFrequency"] = 7
    df_csv["ClickAndCollectActive"] = False

    # --- Cálculo de días ---
    mapa_dias = {
        "LUNES": 1, "MARTES": 2, "MIERCOLES": 3,
        "JUEVES": 4, "VIERNES": 5, "SABADO": 6, "DOMINGO": 7
    }
    df_filtrado["dia_num"] = df_filtrado["dia_visita"].map(mapa_dias)
    df_filtrado["dia_sumado"] = df_filtrado["dia_num"] + 1
    df_filtrado.loc[df_filtrado["dia_sumado"] >= 7, "dia_sumado"] = 1

    df_dias = df_filtrado.groupby("COD CLIENTE BEES")["dia_sumado"].apply(list).reset_index()

    mapa_columnas = {
        1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu",
        5: "Fri", 6: "Sat", 7: "Sun"
    }

    df_csv.set_index("Document", inplace=True)
    for _, row in df_dias.iterrows():
        doc = row["COD CLIENTE BEES"]
        dias = row["dia_sumado"]
        for d in dias:
            if doc in df_csv.index:
                col = mapa_columnas.get(d)
                if col:
                    df_csv.at[doc, col] = 1
    df_csv.reset_index(inplace=True)

    return df_csv


def generar_csv(df_csv, nombre_archivo):
    """
    Guarda el DataFrame procesado como CSV y devuelve el nombre de archivo generado.
    Si no se pasa nombre_archivo, se genera uno con formato 'mes ddmmyyyy.csv'.
    """
    if not nombre_archivo:
        fecha_actual = datetime.now()
        nombre_archivo = f"{fecha_actual.strftime('%B').lower()} {fecha_actual.strftime('%d%m%Y')}.csv"
    if not nombre_archivo.lower().endswith(".csv"):
        nombre_archivo += ".csv"
    df_csv.to_csv(nombre_archivo, index=False, encoding="utf-8-sig")

    return nombre_archivo
