import json
import urllib.request
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st
from transformers import TrainerCallback

from taires.ml.inference import TirePredictor

try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    ClientError = Exception

# --- DIRECTORY CONFIGURATION ---
VERSIONS_DIR = Path("versions")
UPLOAD_DIR = Path("uploads")
VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# --- CLOUD CONFIGURATION ---
S3_CONFIG_FILE = Path(".s3_config.json")


def load_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def load_s3_config() -> dict[str, str]:
    """Load S3 credentials from a local hidden file."""
    if S3_CONFIG_FILE.is_file():
        try:
            return json.loads(S3_CONFIG_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def save_s3_config(config: dict[str, str]) -> None:
    """Persist S3 credentials locally."""
    S3_CONFIG_FILE.write_text(json.dumps(config, indent=4), encoding="utf-8")


def upload_to_s3(file_path: Path, config: dict[str, str], target_key: str) -> None:
    """Execute file upload to S3-compatible storage (e.g., Yandex Cloud)."""
    if not BOTO3_AVAILABLE:
        raise ImportError("boto3 is required but not installed.")
        
    session = boto3.session.Session()
    client = session.client(
        service_name="s3",
        endpoint_url=config.get("endpoint_url", "https://storage.yandexcloud.net"),
        aws_access_key_id=config.get("access_key"),
        aws_secret_access_key=config.get("secret_key"),
        region_name=config.get("region", "ru-central1"),
    )
    client.upload_file(str(file_path), config.get("bucket"), target_key)


def find_version_archive(version_dir: Path) -> Path | None:
    """Locate the packaged archive corresponding to a version directory."""
    for ext in [".zip", ".tar", ".tar.gz"]:
        potential_path = version_dir.parent / f"{version_dir.name}{ext}"
        if potential_path.is_file():
            return potential_path
    return None


@st.cache_resource(show_spinner=False)
def get_cached_predictor(model_dir_str: str) -> TirePredictor:
    """Cache model instance to prevent VRAM reallocation."""
    return TirePredictor(Path(model_dir_str))


@st.cache_data(ttl=3600)
def search_huggingface_models(query: str) -> list[str]:
    if len(query) < 3:
        return []
    url = f"https://huggingface.co/api/models?search={query}&limit=10"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 tAIres/1.0"})
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            return [model["modelId"] for model in data]
    except Exception as e:  # noqa: BLE001
        st.warning(f"Hugging Face API Error: {e}")
        return []


def render_evaluation_curves(curves: dict[str, Any]) -> None:
    c1, c2 = st.columns(2)
    roc, pr = curves.get("roc", {}), curves.get("precision_recall", {})

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=roc.get("fpr", []), y=roc.get("tpr", []), name="ROC", line={"color": "#10B981"}))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random", line={"dash": "dash", "color": "#6B7280"}))
    fig_roc.update_layout(title="ROC Curve", margin={"l": 0, "r": 0, "t": 30, "b": 0}, height=300)
    c1.plotly_chart(fig_roc, width="stretch")

    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=pr.get("recall", []), y=pr.get("precision", []), name="PR", line={"color": "#F59E0B"}))
    fig_pr.update_layout(title="PR Curve", margin={"l": 0, "r": 0, "t": 30, "b": 0}, height=300)
    c2.plotly_chart(fig_pr, width="stretch")


class StreamlitTrainingCallback(TrainerCallback):
    """Stream training progress and logs to the UI."""
    def __init__(self, progress_container: Any, log_container: Any) -> None:
        self.progress_container = progress_container
        self.log_container = log_container
        self.log_history: list[str] = []

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if state.max_steps > 0:
            pct = min(state.global_step / state.max_steps, 1.0)
            self.progress_container.progress(pct, text=f"Training: step {state.global_step} of {state.max_steps}")

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        if logs:
            clean_logs = {k: round(v, 4) if isinstance(v, float) else v for k, v in logs.items()}
            self.log_history.append(f"Step {state.global_step}: {clean_logs}")
            self.log_container.code("\n".join(self.log_history[-12:]), language="json")