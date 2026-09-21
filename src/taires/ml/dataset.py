import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class PairDataset(Dataset):
    """Dataset for text matching optimized for dynamic padding."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_len: int = 64,
    ) -> None:
        self.texts1: list[str] = df["text1"].astype(str).tolist()
        self.texts2: list[str] = df["text2"].astype(str).tolist()
        self.labels: list[int] = df["label"].astype(int).tolist()
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.max_len: int = max_len

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts1[idx],
            self.texts2[idx],
            truncation=True,
            padding=False,
            max_length=self.max_len,
            return_tensors="pt",
        )
        item: dict[str, torch.Tensor] = {
            key: val.squeeze(0) for key, val in encoding.items()
        }
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item