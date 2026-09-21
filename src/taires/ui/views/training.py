import streamlit as st

from taires.ml.create_version import create_model_version
from taires.schemas.training import TrainConfig
from taires.ui.common import (
    UPLOAD_DIR,
    VERSIONS_DIR,
    StreamlitTrainingCallback,
    search_huggingface_models,
)

st.title("Train New Model")

local_models = [str(d / "model") for d in VERSIONS_DIR.iterdir() if (d / "model").exists()]

st.subheader("1. Base Model Configuration")
c1_model, c2_model = st.columns([1, 2])
search_query = c1_model.text_input("Search Hugging Face:", placeholder="3+ characters (e.g., rubert)")

found_models = search_huggingface_models(search_query) if len(search_query) >= 3 else [
    "cointegrated/rubert-tiny2", "DeepPavlov/rubert-base-cased"
]
options = local_models + ["---"] + found_models
base_model = c2_model.selectbox(
    "Transformer for fine-tuning:",
    options=options,
    index=len(local_models) + 1 if local_models else 0,
)

st.write("")

with st.form("train_form", border=False):
    st.subheader("2. Version Context")
    c_meta1, c_meta2 = st.columns(2)
    name = c_meta1.text_input("Name", "rubert-tire-matcher")
    version = c_meta1.text_input("Version", "1.0.0")
    dataset = c_meta2.file_uploader("CSV Dataset", type=["csv"])
    desc = st.text_area("Description", "Training on synthetic data", height=68)

    st.divider()

    st.subheader("3. Hyperparameters")
    c_hyp1, c_hyp2 = st.columns(2)

    with c_hyp1:
        epochs = st.number_input("Epochs", 1, 20, 3)
        batch = st.number_input("Batch Size", 4, 128, 32)
        lr = st.number_input("Learning Rate", value=5e-5, format="%.6f")

    with c_hyp2:
        arch = st.selectbox("Archive Format", ["zip", "tar"])
        st.markdown("<br>", unsafe_allow_html=True)
        clean = st.checkbox("Remove unpacked folder after archiving", value=False)

    st.write("")
    submitted = st.form_submit_button("Run Training Pipeline", width="stretch")

if submitted:
    if not dataset:
        st.error("Please upload a CSV dataset.")
    elif base_model == "---":
        st.error("Please select a valid transformer from the list.")
    else:
        data_file = UPLOAD_DIR / dataset.name
        data_file.write_bytes(dataset.getbuffer())

        st.write("### Training Progress")
        st_progress = st.progress(0, text="Initializing pipeline...")
        st_logs = st.empty()
        st_callback = StreamlitTrainingCallback(st_progress, st_logs)

        with st.spinner(f"Training on base model {base_model}..."):
            try:
                cfg = TrainConfig(
                    data_path=data_file,
                    model_name=base_model,
                    num_train_epochs=epochs,
                    per_device_train_batch_size=batch,
                    learning_rate=lr,
                )
                out = create_model_version(
                    config=cfg,
                    name=name,
                    version=version,
                    description=desc,
                    archive_format=arch,
                    remove_unpacked=clean,
                    callbacks=[st_callback],
                )
                st_progress.progress(1.0, text="Training completed successfully!")
                st.success(f"Successfully packaged: {out.resolve()}")
            except Exception as e:  # noqa: BLE001
                st.error(f"Build error: {e}")
            finally:
                if data_file.exists():
                    data_file.unlink()