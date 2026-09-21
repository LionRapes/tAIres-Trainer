from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class TirePredictor:
    """Inference engine for versioned cross-encoder checkpoints."""

    def __init__(self, model_dir: Path, device: str | None = None) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_pair(self, text1: str, text2: str) -> float:
        """Calculate match confidence for a single pair of descriptions."""
        inputs = self.tokenizer(
            text1,
            text2,
            truncation=True,
            padding=True,
            max_length=64,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[:, 1]
        return float(probs.cpu().item())