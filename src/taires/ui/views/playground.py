import streamlit as st

from taires.ui.common import VERSIONS_DIR, get_cached_predictor, load_json

st.title("Inference Playground")
dirs = [d for d in VERSIONS_DIR.iterdir() if (d / "model").is_dir()]

if not dirs:
    st.warning("No available models for inference.")
else:
    sel_dir = st.selectbox("Active model:", dirs, format_func=lambda x: x.name)
    meta = load_json(sel_dir / "metadata.json")
    thresh = meta.get("decision_threshold", 0.5) if meta else 0.5

    c1, c2 = st.columns(2)
    t1 = c1.text_area("String 1 (Supplier):", "A")
    t2 = c2.text_area("String 2 (Catalog):", "B")

    if st.button("Compare Strings", width="stretch"):
        with st.spinner("Running inference..."):
            predictor = get_cached_predictor(str(sel_dir / "model"))
            score = predictor.predict_pair(t1, t2)

            st.write("")
            rc1, rc2 = st.columns([1, 3])
            rc1.metric("Confidence", f"{score * 100:.1f}%")
            if score >= thresh:
                rc2.success(f"MATCH (Threshold: {thresh:.3f})")
            else:
                rc2.error(f"MISMATCH (Threshold: {thresh:.3f})")