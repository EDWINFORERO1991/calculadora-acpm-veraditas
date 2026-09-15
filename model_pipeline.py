import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor

ID_HOJA = "14ZwkJ_NtBiTvPV4IGi9HNoX9qWva_xUH"
URL_DATASET = f"https://docs.google.com/spreadsheets/d/{ID_HOJA}/export?format=xlsx"

VARIABLES_NUMERICAS = ["HP", "HORAS_TRABAJO_CICLO"]
VARIABLES_CATEGORICAS = ["ACTIVIDAD", "IMPLEMENTO", "EQUIPO/VEHÍCULO"]
VARIABLES_PREDICTORAS = [
    "HP",
    "HORAS_TRABAJO_CICLO",
    "ACTIVIDAD",
    "IMPLEMENTO",
    "EQUIPO/VEHÍCULO",
]
VARIABLE_OBJETIVO = "ACPM_ASOCIADO_CICLO"


def cargar_dataset():
    df = pd.read_excel(URL_DATASET)

    df_limpio = df.copy()
    df_limpio["FECHA EJECUCIÓN ACTIVIDAD"] = pd.to_datetime(
        df_limpio["FECHA EJECUCIÓN ACTIVIDAD"],
        errors="coerce"
    )

    columnas_numericas = [
        "HP",
        "HOROMETRO",
        "ABASTECIMIENTO ACPM\n(Galones)",
        "HORAS DE TRABAJO",
    ]

    for columna in columnas_numericas:
        df_limpio[columna] = (
            df_limpio[columna]
            .astype(str)
            .str.replace(",", "", regex=False)
            .replace("nan", np.nan)
        )
        df_limpio[columna] = pd.to_numeric(
            df_limpio[columna],
            errors="coerce"
        )

    # Igual que el notebook: un registro es tractor cuando tiene HP.
    df_limpio["ES_TRACTOR"] = df_limpio["HP"].notna()
    df_tractores = df_limpio[df_limpio["ES_TRACTOR"]].copy()

    # Orden cronológico.
    df_reconstruccion = df_tractores.copy()
    df_reconstruccion["ORDEN_REGISTRO"] = df_reconstruccion.index
    df_reconstruccion = df_reconstruccion.sort_values(
        ["EQUIPO/VEHÍCULO", "FECHA EJECUCIÓN ACTIVIDAD", "ORDEN_REGISTRO"]
    ).reset_index(drop=True)

    col_acpm = "ABASTECIMIENTO ACPM\n(Galones)"
    df_ciclos = df_reconstruccion.copy()

    df_ciclos["ES_ABASTECIMIENTO"] = (
        df_ciclos[col_acpm].fillna(0) > 0
    )

    # El abastecimiento cierra el período anterior.
    df_ciclos["CICLO"] = (
        df_ciclos
        .groupby("EQUIPO/VEHÍCULO")["ES_ABASTECIMIENTO"]
        .cumsum()
        - df_ciclos["ES_ABASTECIMIENTO"].astype(int)
    )

    # Resumen de ciclos.
    df_resumen_ciclos = (
        df_ciclos
        .groupby(["EQUIPO/VEHÍCULO", "CICLO"], as_index=False)
        .agg(
            FECHA_INICIAL=("FECHA EJECUCIÓN ACTIVIDAD", "min"),
            FECHA_FINAL=("FECHA EJECUCIÓN ACTIVIDAD", "max"),
            REGISTROS=("EQUIPO/VEHÍCULO", "size"),
            HORAS_TRABAJO=("HORAS DE TRABAJO", "sum"),
            ACPM_ASOCIADO_CICLO=(
                col_acpm,
                lambda x: x[x > 0].sum()
            ),
        )
    )

    # Homogeneidad.
    actividades_por_ciclo = (
        df_ciclos
        .groupby(["EQUIPO/VEHÍCULO", "CICLO"])["LABOR MÁQUINA/EQUIPO"]
        .nunique(dropna=True)
        .reset_index(name="NUM_ACTIVIDADES")
    )

    implementos_por_ciclo = (
        df_ciclos
        .groupby(["EQUIPO/VEHÍCULO", "CICLO"])["IMPLEMENTO/EQUIPO AUXILIAR"]
        .nunique(dropna=True)
        .reset_index(name="NUM_IMPLEMENTOS")
    )

    resumen_homogeneidad = actividades_por_ciclo.merge(
        implementos_por_ciclo,
        on=["EQUIPO/VEHÍCULO", "CICLO"],
        how="outer"
    )

    resumen_homogeneidad["ACTIVIDAD_UNICA"] = (
        resumen_homogeneidad["NUM_ACTIVIDADES"] == 1
    )
    resumen_homogeneidad["IMPLEMENTO_UNICO"] = (
        resumen_homogeneidad["NUM_IMPLEMENTOS"] == 1
    )

    # Validación de horómetro por ciclo, replicando el criterio del notebook.
    datos_horometro = df_ciclos.copy()
    datos_horometro["HOROMETRO"] = pd.to_numeric(
        datos_horometro["HOROMETRO"], errors="coerce"
    )
    datos_horometro["HORAS DE TRABAJO"] = pd.to_numeric(
        datos_horometro["HORAS DE TRABAJO"], errors="coerce"
    )

    datos_horometro = datos_horometro.sort_values(
        ["EQUIPO/VEHÍCULO", "FECHA EJECUCIÓN ACTIVIDAD", "ORDEN_REGISTRO"]
    ).copy()

    datos_horometro["HOROMETRO_ANTERIOR"] = (
        datos_horometro
        .groupby("EQUIPO/VEHÍCULO")["HOROMETRO"]
        .shift(1)
    )

    datos_horometro["DIFERENCIA_HOROMETRO"] = (
        datos_horometro["HOROMETRO"]
        - datos_horometro["HOROMETRO_ANTERIOR"]
    )

    datos_horometro["DISMINUCION_HOROMETRO"] = (
        datos_horometro["DIFERENCIA_HOROMETRO"] < 0
    )

    def resumir_horometro_ciclo(grupo):
        diferencias = grupo["DIFERENCIA_HOROMETRO"]
        horas_horometro = diferencias.sum(min_count=1)
        horas_trabajo = grupo["HORAS DE TRABAJO"].sum(min_count=1)

        if pd.notna(horas_horometro) and pd.notna(horas_trabajo):
            diferencia = horas_horometro - horas_trabajo
        else:
            diferencia = np.nan

        return pd.Series({
            "HORAS_HOROMETRO": horas_horometro,
            "HORAS_TRABAJO_CICLO": horas_trabajo,
            "DIFERENCIA_HORAS": diferencia,
            "DISMINUCIONES_HOROMETRO": int(
                grupo["DISMINUCION_HOROMETRO"].sum()
            ),
        })

    validation_horometro = (
        datos_horometro
        .groupby(["EQUIPO/VEHÍCULO", "CICLO"], sort=False)
        .apply(resumir_horometro_ciclo, include_groups=False)
        .reset_index()
    )

    validation_horometro["HOROMETRO_CONSISTENTE"] = (
        validation_horometro["DIFERENCIA_HORAS"].abs() <= 0.10
    )

    # Integración de los tres componentes de confiabilidad.
    hom = resumen_homogeneidad[
        ["EQUIPO/VEHÍCULO", "CICLO", "ACTIVIDAD_UNICA", "IMPLEMENTO_UNICO"]
    ].copy()

    val = validation_horometro[
        [
            "EQUIPO/VEHÍCULO",
            "CICLO",
            "HOROMETRO_CONSISTENTE",
            "DISMINUCIONES_HOROMETRO",
        ]
    ].copy()

    base = df_resumen_ciclos[
        [
            "EQUIPO/VEHÍCULO",
            "CICLO",
            "FECHA_INICIAL",
            "FECHA_FINAL",
            "REGISTROS",
            "ACPM_ASOCIADO_CICLO",
            "HORAS_TRABAJO",
        ]
    ].rename(columns={"HORAS_TRABAJO": "HORAS_TRABAJO_CICLO"})

    df_analitico = (
        base
        .merge(hom, on=["EQUIPO/VEHÍCULO", "CICLO"], how="left")
        .merge(val, on=["EQUIPO/VEHÍCULO", "CICLO"], how="left")
    )

    df_analitico["CRITERIO_ACTIVIDAD"] = (
        df_analitico["ACTIVIDAD_UNICA"] == True
    )
    df_analitico["CRITERIO_IMPLEMENTO"] = (
        df_analitico["IMPLEMENTO_UNICO"] == True
    )
    df_analitico["CRITERIO_ACPM"] = (
        pd.to_numeric(
            df_analitico["ACPM_ASOCIADO_CICLO"], errors="coerce"
        ) > 0
    )
    df_analitico["CRITERIO_HORAS"] = (
        pd.to_numeric(
            df_analitico["HORAS_TRABAJO_CICLO"], errors="coerce"
        ) > 0
    )
    df_analitico["CRITERIO_HOROMETRO"] = (
        df_analitico["HOROMETRO_CONSISTENTE"] == True
    )
    df_analitico["CRITERIO_SIN_DISMINUCION"] = (
        pd.to_numeric(
            df_analitico["DISMINUCIONES_HOROMETRO"], errors="coerce"
        ).fillna(0) == 0
    )

    criterios = [
        "CRITERIO_ACTIVIDAD",
        "CRITERIO_IMPLEMENTO",
        "CRITERIO_ACPM",
        "CRITERIO_HORAS",
        "CRITERIO_HOROMETRO",
        "CRITERIO_SIN_DISMINUCION",
    ]

    df_analitico["ALTA_CONFIABILIDAD"] = (
        df_analitico[criterios].all(axis=1)
    )

    df_modelo = df_analitico[
        df_analitico["ALTA_CONFIABILIDAD"]
    ].copy()

    # HP por equipo, igual que en el notebook.
    hp_lookup = (
        df_tractores[
            ["EQUIPO/VEHÍCULO", "HP"]
        ]
        .dropna(subset=["HP"])
        .drop_duplicates(subset=["EQUIPO/VEHÍCULO"])
    )

    df_modelo = df_modelo.merge(
        hp_lookup,
        on="EQUIPO/VEHÍCULO",
        how="left"
    )

    # Actividad e implemento representativos del ciclo.
    datos_operativos = df_ciclos[
        [
            "EQUIPO/VEHÍCULO",
            "CICLO",
            "LABOR MÁQUINA/EQUIPO",
            "IMPLEMENTO/EQUIPO AUXILIAR",
        ]
    ].copy()

    actividad_implemento = (
        datos_operativos
        .groupby(["EQUIPO/VEHÍCULO", "CICLO"], as_index=False)
        .agg(
            ACTIVIDAD=("LABOR MÁQUINA/EQUIPO", "first"),
            IMPLEMENTO=("IMPLEMENTO/EQUIPO AUXILIAR", "first"),
        )
    )

    df_modelo = (
        df_modelo
        .merge(
            actividad_implemento,
            on=["EQUIPO/VEHÍCULO", "CICLO"],
            how="left"
        )
    )

    df_modelo["CONSUMO_GAL_HORA"] = (
        df_modelo["ACPM_ASOCIADO_CICLO"]
        / df_modelo["HORAS_TRABAJO_CICLO"]
    )

    # Criterio de valores extremos del notebook:
    # límite superior IQR calculado sobre CONSUMO_GAL_HORA.
    q1 = df_modelo["CONSUMO_GAL_HORA"].quantile(0.25)
    q3 = df_modelo["CONSUMO_GAL_HORA"].quantile(0.75)
    iqr = q3 - q1
    limite_superior = q3 + 1.5 * iqr

    df_modelo["ES_EXTREMO"] = (
        df_modelo["CONSUMO_GAL_HORA"] > limite_superior
    )

    df_modelado = (
        df_modelo[
            ~df_modelo["ES_EXTREMO"]
        ]
        .copy()
        .reset_index(drop=True)
    )

    return df_modelado


def train_model():
    df_modelado = cargar_dataset()

    # El notebook utiliza exactamente estas variables.
    X = df_modelado[VARIABLES_PREDICTORAS].copy()
    y = df_modelado[VARIABLE_OBJETIVO].copy()

    preprocesador = ColumnTransformer(
        transformers=[
            (
                "numericas",
                "passthrough",
                VARIABLES_NUMERICAS,
            ),
            (
                "categoricas",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                VARIABLES_CATEGORICAS,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocesamiento", preprocesador),
            (
                "modelo",
                GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42,
                ),
            ),
        ]
    )

    # Para una aplicación predictiva se entrena con todo el conjunto
    # analítico disponible después del filtro de extremos.
    model.fit(X, y)

    tractors = sorted(
        df_modelado["EQUIPO/VEHÍCULO"].dropna().unique().tolist()
    )

    hp_map = (
        df_modelado[["EQUIPO/VEHÍCULO", "HP"]]
        .drop_duplicates("EQUIPO/VEHÍCULO")
        .set_index("EQUIPO/VEHÍCULO")["HP"]
        .to_dict()
    )

    activities = sorted(
        df_modelado["ACTIVIDAD"].dropna().unique().tolist()
    )

    implements_by_activity = (
        df_modelado
        .dropna(subset=["ACTIVIDAD", "IMPLEMENTO"])
        .groupby("ACTIVIDAD")["IMPLEMENTO"]
        .apply(lambda s: sorted(s.unique().tolist()))
        .to_dict()
    )

    catalog = (tractors, hp_map, activities, implements_by_activity)

    return model, catalog


def get_catalogs():
    _, catalog = train_model()
    return catalog
