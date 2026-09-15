import streamlit as st
import pandas as pd
import numpy as np

from model_pipeline import train_model, get_catalogs

st.set_page_config(
    page_title="Calculadora de Consumo de ACPM",
    page_icon="🚜",
    layout="wide",
)

# -----------------------------
# Estilos
# -----------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
    }
    .subtitle {
        color: #5f6b7a;
        margin-bottom: 1.2rem;
    }
    .result-card {
        padding: 0.65rem 0.8rem;
        border-radius: 10px;
        background: #eef7ff;
        border: 1px solid #cfe5f7;
        text-align: center;
        font-weight: 700;
    }
    .total-card {
        padding: 1rem;
        border-radius: 12px;
        background: #0b2d4d;
        color: white;
        text-align: center;
    }
    .small-note {
        color: #667085;
        font-size: 0.86rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🚜 Calculadora de Consumo de ACPM</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Predicción del consumo asociado a ciclos operativos de maquinaria agrícola.</div>',
    unsafe_allow_html=True
)

@st.cache_resource(show_spinner="Preparando el modelo predictivo…")
def load_model():
    return train_model()

try:
    model, catalog = load_model()
except Exception as e:
    st.error("No fue posible preparar el modelo con el dataset configurado.")
    st.exception(e)
    st.stop()

tractors, hp_map, activities, implements_by_activity = catalog

st.info(
    "El modelo utiliza las mismas variables del notebook: HP, horas de trabajo del ciclo, "
    "actividad, implemento y equipo/vehículo. El modelo seleccionado es Gradient Boosting."
)

st.markdown("### Ingreso de operaciones")

with st.form("form_predicciones"):
    rows = []

    header = st.columns([0.04, 0.22, 0.09, 0.27, 0.27, 0.11])
    header[0].markdown("**#**")
    header[1].markdown("**Tractor**")
    header[2].markdown("**HP**")
    header[3].markdown("**Actividad**")
    header[4].markdown("**Implemento**")
    header[5].markdown("**Horas**")

    for i in range(10):
        cols = st.columns([0.04, 0.22, 0.09, 0.27, 0.27, 0.11])

        with cols[0]:
            st.write(str(i + 1))

        with cols[1]:
            tractor = st.selectbox(
                "Tractor",
                ["— Seleccionar —"] + tractors,
                key=f"tractor_{i}",
                label_visibility="collapsed",
            )

        hp = hp_map.get(tractor, np.nan)
        with cols[2]:
            if pd.notna(hp):
                st.number_input(
                    "HP",
                    value=float(hp),
                    disabled=True,
                    key=f"hp_{i}",
                    label_visibility="collapsed",
                )
            else:
                st.write("—")

        with cols[3]:
            activity_options = ["— Seleccionar —"] + activities
            activity = st.selectbox(
                "Actividad",
                activity_options,
                key=f"activity_{i}",
                label_visibility="collapsed",
            )

        with cols[4]:
            if activity != "— Seleccionar —":
                impl_options = ["— Seleccionar —"] + implements_by_activity.get(activity, [])
            else:
                impl_options = ["— Seleccionar —"] + sorted(set(implements_by_activity.get(a, []) for a in []))
                # Sin actividad: se muestra el catálogo completo.
                all_impl = sorted(set(x for values in implements_by_activity.values() for x in values))
                impl_options = ["— Seleccionar —"] + all_impl

            implement = st.selectbox(
                "Implemento",
                impl_options,
                key=f"implement_{i}",
                label_visibility="collapsed",
            )

        with cols[5]:
            hours = st.number_input(
                "Horas",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=0.1,
                key=f"hours_{i}",
                label_visibility="collapsed",
            )

        rows.append({
            "fila": i + 1,
            "tractor": tractor,
            "hp": hp,
            "actividad": activity,
            "implemento": implement,
            "horas": hours,
        })

    submitted = st.form_submit_button("CALCULAR CONSUMO", type="primary", use_container_width=True)

if submitted:
    valid_rows = []
    warnings = []

    for r in rows:
        if r["tractor"] == "— Seleccionar —" and r["activity"] == "— Seleccionar —" and r["implemento"] == "— Seleccionar —" and r["horas"] == 0:
            continue

        missing = []
        if r["tractor"] == "— Seleccionar —":
            missing.append("tractor")
        if r["actividad"] == "— Seleccionar —":
            missing.append("actividad")
        if r["implemento"] == "— Seleccionar —":
            missing.append("implemento")
        if r["horas"] <= 0:
            missing.append("horas")

        if missing:
            warnings.append(f"Fila {r['fila']}: faltan " + ", ".join(missing))
            continue

        valid_rows.append(r)

    if warnings:
        for w in warnings:
            st.warning(w)

    if valid_rows:
        X = pd.DataFrame([{
            "HP": r["hp"],
            "HORAS_TRABAJO_CICLO": r["horas"],
            "ACTIVIDAD": r["actividad"],
            "IMPLEMENTO": r["implemento"],
            "EQUIPO/VEHÍCULO": r["tractor"],
        } for r in valid_rows])

        preds = np.maximum(model.predict(X), 0)

        results = []
        for r, pred in zip(valid_rows, preds):
            results.append({
                "Fila": r["fila"],
                "Tractor": r["tractor"],
                "HP": int(r["hp"]),
                "Actividad": r["actividad"],
                "Implemento": r["implemento"],
                "Horas": r["hours"],
                "ACPM estimado (gal)": round(float(pred), 2),
            })

        result_df = pd.DataFrame(results)

        st.markdown("### Resultados")

        display_df = result_df.copy()
        display_df["ACPM estimado (gal)"] = display_df["ACPM estimado (gal)"].map(lambda x: f"{x:,.2f}")
        display_df["Horas"] = display_df["Horas"].map(lambda x: f"{x:,.1f}")

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        total_gal = float(result_df["ACPM estimado (gal)"].sum())
        total_h = float(result_df["Horas"].sum())
        avg = total_gal / total_h if total_h > 0 else 0

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Operaciones calculadas", len(result_df))
        with c2:
            st.metric("Horas totales", f"{total_h:,.1f} h")
        with c3:
            st.metric("ACPM total estimado", f"{total_gal:,.2f} gal")

        st.caption(f"Consumo promedio ponderado: {avg:,.2f} gal/h")

        st.download_button(
            "Descargar resultados CSV",
            data=result_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="prediccion_consumo_acpm.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.markdown("---")
st.markdown(
    '<div class="small-note">'
    "Fuente metodológica: notebook Forero_Edwin_Actividad5.ipynb. "
    "La variable objetivo es ACPM_ASOCIADO_CICLO, expresada en galones. "
    "El modelo Gradient Boosting fue seleccionado según el MAE del conjunto de prueba."
    "</div>",
    unsafe_allow_html=True
)
