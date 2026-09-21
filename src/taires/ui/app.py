import streamlit as st

st.set_page_config(
    page_title="tAIres Studio",
    layout="wide",
)

pg = st.navigation(
    [
        st.Page("views/registry.py", title="Model Registry", icon=":material/database:", default=True),
        st.Page("views/playground.py", title="Inference Playground", icon=":material/science:"),
        st.Page("views/training.py", title="Train New Model", icon=":material/tune:"),
    ]
)
pg.run()