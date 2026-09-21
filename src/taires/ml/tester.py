import logging
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from taires.schemas.training import AutomatedTestsReport, TestSampleResult

logger = logging.getLogger(__name__)


def extract_test_suite_from_dataset(
    dataset_path: str | Path,
    samples_per_class: int = 15,
    random_state: int = 42,
) -> list[dict]:
    """Automatically sample balanced test cases from a CSV dataset."""
    df = pd.read_csv(dataset_path)

    required_cols = {"text1", "text2", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Dataset requires {required_cols} columns. Found: {list(df.columns)}"
        )

    sampled_rows: list[pd.DataFrame] = []

    if "pair_type" in df.columns and not df["pair_type"].isna().all():
        for (_, _), group in df.groupby(["pair_type", "label"], as_index=False):
            n_samples = min(len(group), max(1, samples_per_class // 3))
            sampled_rows.append(group.sample(n=n_samples, random_state=random_state))
    else:
        for _, group in df.groupby("label", as_index=False):
            n_samples = min(len(group), samples_per_class)
            sampled_rows.append(group.sample(n=n_samples, random_state=random_state))

    if not sampled_rows:
        raise ValueError("Could not extract samples from the dataset.")

    sampled_df = pd.concat(sampled_rows, ignore_index=True)

    test_suite = [
        {
            "text1": str(row["text1"]),
            "text2": str(row["text2"]),
            "expected": bool(row["label"] == 1),
        }
        for row in sampled_df[["text1", "text2", "label"]].to_dict('records')
    ]
    logger.info(
        "Extracted %d automated test cases from %s", len(test_suite), dataset_path
    )
    return test_suite


def run_automated_tests(
    model_dir: Path,
    dataset_path: str | Path | None = None,
    test_suite: list[dict] | None = None,
    threshold: float = 0.5,
    samples_per_class: int = 20,
    batch_size: int = 16,
) -> AutomatedTestsReport:
    """Run batched evaluation tests against generated or provided test suites."""
    if test_suite is None:
        if dataset_path is None:
            raise ValueError("Either 'test_suite' or 'dataset_path' must be provided.")
        test_suite = extract_test_suite_from_dataset(
            dataset_path=dataset_path,
            samples_per_class=samples_per_class,
        )

    if not test_suite:
        raise ValueError("Test suite is empty.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    results: list[TestSampleResult] = []
    passed_count = 0

    with torch.no_grad():
        for i in range(0, len(test_suite), batch_size):
            batch_cases = test_suite[i : i + batch_size]
            texts1 = [c["text1"] for c in batch_cases]
            texts2 = [c["text2"] for c in batch_cases]

            inputs = tokenizer(
                texts1,
                texts2,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=64,
            ).to(device)

            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()

            for case, prob in zip(batch_cases, probs):
                confidence = float(prob)
                predicted_match = confidence >= threshold
                passed = predicted_match == case["expected"]

                if passed:
                    passed_count += 1

                results.append(
                    TestSampleResult(
                        text1=case["text1"],
                        text2=case["text2"],
                        expected_match=case["expected"],
                        predicted_match=predicted_match,
                        confidence=round(confidence, 4),
                        passed=passed,
                    )
                )

    total = len(test_suite)
    return AutomatedTestsReport(
        total_tests=total,
        passed_tests=passed_count,
        accuracy=round(passed_count / total, 4) if total > 0 else 0.0,
        results=results,
    )