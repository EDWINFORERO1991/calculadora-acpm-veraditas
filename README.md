# Calculadora web de consumo de ACPM

Aplicación Streamlit basada en `Forero_Edwin_Actividad5.ipynb`.

## Qué hace

Permite registrar hasta 10 operaciones en una sola pantalla:

- Tractor (lista desplegable)
- HP automático según el tractor
- Actividad (lista desplegable)
- Implemento (lista desplegable dependiente de la actividad)
- Horas de trabajo

Al presionar **CALCULAR CONSUMO**, la aplicación entrega el ACPM estimado por operación y el total.

## Modelo

Se utiliza el mismo Gradient Boosting definido en el notebook:

- `n_estimators=200`
- `learning_rate=0.05`
- `max_depth=3`
- `random_state=42`

Variables predictoras:

- HP
- HORAS_TRABAJO_CICLO
- ACTIVIDAD
- IMPLEMENTO
- EQUIPO/VEHÍCULO

Variable objetivo:

- `ACPM_ASOCIADO_CICLO`

El notebook seleccionó Gradient Boosting por MAE sobre el conjunto de prueba.

## Fuente de datos

La aplicación lee la hoja de Google Sheets configurada en `model_pipeline.py`.

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publicación

Este proyecto está preparado para desplegarse en un servicio compatible con Streamlit. Una vez publicado se obtiene un enlace web que puede abrirse desde computador o celular sin instalar Python en el equipo del usuario.

## Nota metodológica

Para la aplicación, el modelo se reentrena con todo el conjunto analítico disponible después del mismo proceso de construcción y filtro de valores extremos. Esto es apropiado para utilizar el modelo como predictor final, pero no cambia los resultados reportados en el notebook sobre el conjunto de prueba.
