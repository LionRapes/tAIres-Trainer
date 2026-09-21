import pandas as pd
import plotly.express as px
import streamlit as st

from taires.ui.common import (
    BOTO3_AVAILABLE,
    VERSIONS_DIR,
    ClientError,
    find_version_archive,
    load_json,
    load_s3_config,
    render_evaluation_curves,
    save_s3_config,
    upload_to_s3,
)

st.title("Model Registry")
versions = [d.name for d in VERSIONS_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")]

if not versions:
    st.info("No trained models found in the `versions/` directory.")
    st.stop()

selected_dir = VERSIONS_DIR / st.selectbox("Select version:", versions)
meta = load_json(selected_dir / "metadata.json")
tlm = load_json(selected_dir / "telemetry.json")

if meta:
    st.markdown(f"**Description:** {meta.get('description')}")
    metrics = meta.get("metrics", {})

    cols = st.columns(5)
    keys = [
        ("ROC-AUC", "test_roc_auc"),
        ("PR-AUC", "test_pr_auc"),
        ("F1 Score", "test_f1"),
        ("Accuracy", "test_accuracy"),
        ("Threshold", "optimal_threshold"),
    ]
    for col, (label, key) in zip(cols, keys):
        val = metrics.get(key, meta.get("decision_threshold") if key == "optimal_threshold" else 0)
        col.metric(label, f"{val:.4f}")

    t1, t2, t3 = st.tabs(["Curves", "Confusion Matrix", "Automated Tests"])

    with t1:
        if tlm and "curves" in tlm:
            render_evaluation_curves(tlm["curves"])
    with t2:
        if tlm and "confusion_matrix" in tlm:
            fig = px.imshow(
                tlm["confusion_matrix"],
                text_auto=True,
                color_continuous_scale="Blues",
                x=["Mismatch", "Match"],
                y=["Mismatch", "Match"],
            )
            st.plotly_chart(fig.update_layout(height=400, margin={"t": 0}))
    with t3:
        tests = load_json(selected_dir / "test_results.json")
        if tests:
            accuracy_pct = tests.get("accuracy", 0) * 100
            st.success(f"Passed: {tests.get('passed_tests')}/{tests.get('total_tests')} ({accuracy_pct:.1f}%)")
            st.dataframe(pd.DataFrame(tests.get("results", [])), width="stretch")

    st.divider()
    
    # --- DEPLOYMENT WIDGET ---
    st.subheader("Deployment")
    
    if not BOTO3_AVAILABLE:
        st.error("The `boto3` library is required for deployment. Run `poetry add boto3`.")
    else:
        with st.expander("Deploy to Object Storage (S3)", expanded=False):
            s3_cfg = load_s3_config()
            archive_file = find_version_archive(selected_dir)
            
            with st.form("s3_deploy_form", border=False):
                c_s3_1, c_s3_2 = st.columns(2)
                
                endpoint = c_s3_1.text_input("Endpoint URL", value=s3_cfg.get("endpoint_url", "https://storage.yandexcloud.net"))
                bucket = c_s3_2.text_input("Bucket Name", value=s3_cfg.get("bucket", ""))
                
                access_key = c_s3_1.text_input("Access Key ID", value=s3_cfg.get("access_key", ""))
                secret_key = c_s3_2.text_input("Secret Access Key", value=s3_cfg.get("secret_key", ""), type="password")
                
                default_target = f"models/{archive_file.name}" if archive_file else f"models/{selected_dir.name}.zip"
                target_path = st.text_input("Target Path in Bucket", value=default_target)
                
                deploy_submitted = st.form_submit_button("Save Credentials & Deploy", width="stretch")
                
                if deploy_submitted:
                    new_s3_cfg = {
                        "endpoint_url": endpoint,
                        "bucket": bucket,
                        "access_key": access_key,
                        "secret_key": secret_key,
                        "region": "ru-central1"
                    }
                    save_s3_config(new_s3_cfg)
                            
                    if not archive_file:
                        st.error(f"Archive not found for '{selected_dir.name}'. Verify that the model was packaged during training.")
                    elif not all([endpoint, bucket, access_key, secret_key, target_path]):
                        st.error("All S3 configuration fields are required.")
                    else:
                        if not target_path.endswith(archive_file.suffix):
                            target_path += archive_file.suffix

                        with st.spinner("Uploading artifact to S3..."):
                            try:
                                upload_to_s3(archive_file, new_s3_cfg, target_path)
                                st.success(f"Deployment successful: s3://{bucket}/{target_path}")
                            except ClientError as e:
                                st.error(f"S3 Client Error: {e}")
                            except Exception as e:  # noqa: BLE001
                                st.error(f"Deployment failed: {e}")