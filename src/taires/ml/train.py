import argparse
import gc
import logging
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from taires.ml.dataset import PairDataset
from taires.schemas.training import TrainConfig, TrainingTelemetry
from taires.utils.metrics import build_telemetry_payload, compute_metrics

logger = logging.getLogger(__name__)


def train(config: TrainConfig) -> TrainingTelemetry:
    """Execute model training pipeline using parameters from TrainConfig."""
    if not config.data_path.is_file():
        raise FileNotFoundError(
            f"Provided dataset path does not exist: {config.data_path.resolve()}"
        )

    logger.info("Loading dataset from %s", config.data_path)
    df: pd.DataFrame = pd.read_csv(config.data_path)

    if not {"text1", "text2", "label"}.issubset(df.columns):
        raise ValueError("Dataset must contain 'text1', 'text2', and 'label' columns.")

    if len(df["label"].unique()) < 2:
        raise ValueError("Dataset must contain at least 2 distinct classes.")

    logger.info(
        "Splitting dataset (test_size=%.2f, seed=%d)",
        config.test_size,
        config.random_state,
    )
    train_df, val_df = train_test_split(
        df,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=df["label"],
    )

    config.val_data_path.parent.mkdir(parents=True, exist_ok=True)
    val_df.to_csv(config.val_data_path, index=False)
    logger.info("Validation dataset saved to %s", config.val_data_path)

    config.train_data_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(config.train_data_path, index=False)
    logger.info("Training dataset saved to %s", config.train_data_path)

    logger.info("Initializing tokenizer: %s", config.model_name)
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        config.model_name
    )

    train_dataset = PairDataset(train_df, tokenizer, max_len=config.max_length)
    val_dataset = PairDataset(val_df, tokenizer, max_len=config.max_length)

    logger.info("Loading model: %s", config.model_name)
    model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
        config.model_name, num_labels=2
    )

    config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    use_fp16 = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=str(config.checkpoints_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=20,
        fp16=use_fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=config.logging_steps,
        report_to="none",
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    logger.info("Executing model training...")
    trainer.train()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Persisting best model and tokenizer to %s", config.output_dir)
    trainer.save_model(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))

    logger.info("Evaluating predictions on validation dataset...")
    predictions_output = trainer.predict(val_dataset)

    telemetry: TrainingTelemetry = build_telemetry_payload(
        predictions_output=predictions_output,
        log_history=trainer.state.log_history,
    )

    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return telemetry


def parse_args() -> TrainConfig:
    """Parse CLI arguments with default configuration."""
    parser = argparse.ArgumentParser(
        description="Train sequence classification model."
    )
    parser.add_argument(
        "--data-path", type=Path
    )
    parser.add_argument("--val-data-path", type=Path)
    parser.add_argument(
        "--model-name", type=str, default="DeepPavlov/rubert-base-cased"
    )
    parser.add_argument(
        "--output-dir", type=Path
    )
    parser.add_argument(
        "--checkpoints-dir", type=Path
    )
    parser.add_argument(
        "--epochs", type=int, default=5, dest="num_train_epochs"
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=16,
        dest="per_device_train_batch_size",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=32,
        dest="per_device_eval_batch_size",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5, dest="learning_rate"
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--logging-steps", type=int, default=50)

    args = parser.parse_args()
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    return TrainConfig.model_validate(args_dict)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    train_config = parse_args()
    train(train_config)