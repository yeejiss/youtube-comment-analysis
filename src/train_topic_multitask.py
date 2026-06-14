#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


# =========================
# 0. 라벨 정의 / 기본 함수
# =========================
TARGET_ID2LABEL = {
    0: "Performer",
    1: "Content",
    2: "System",
    3: "Others",
}
ATTRIBUTE_ID2LABEL = {
    0: "Skill",
    1: "Visual",
    2: "Character",
    3: "Production",
    4: "Survival",
    5: "Interpretation",
    6: "None",
}
TARGET_LABEL2ID = {v: k for k, v in TARGET_ID2LABEL.items()}
ATTRIBUTE_LABEL2ID = {v: k for k, v in ATTRIBUTE_ID2LABEL.items()}
NUM_TARGET_CLASSES = len(TARGET_ID2LABEL)
NUM_ATTRIBUTE_CLASSES = len(ATTRIBUTE_ID2LABEL)


def parse_args():
    project_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser()

    # 경로 설정
    parser.add_argument("--data_dir", type=str, default="/local_datasets/yt_topic_labeled_2189")
    parser.add_argument("--output_dir", type=str, default=str(project_root / "results" / "topic_multitask"))
    parser.add_argument("--model_name", type=str, default="beomi/KcELECTRA-base-v2022")
    parser.add_argument("--exp_name", type=str, default="exp")

    # 학습 설정
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--early_stopping_patience", type=int, default=2)

    # 멀티태스크 loss weight
    parser.add_argument("--target_loss_weight", type=float, default=1.0)
    parser.add_argument("--attribute_loss_weight", type=float, default=1.2)

    # best model 선택 기준
    parser.add_argument(
        "--selection_metric",
        type=str,
        default="mean_macro_f1",
        choices=["mean_macro_f1", "joint_acc", "target_macro_f1", "attribute_macro_f1"],
    )

    # 멀티라벨 threshold
    parser.add_argument("--target_threshold", type=float, default=0.48) 
    parser.add_argument("--attribute_threshold", type=float, default=0.33) 


    return parser.parse_args()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# 데이터 로드
def load_split_csv(data_dir: Path, split_name: str) -> pd.DataFrame:
    data_dir = Path(data_dir)
    csv_path = data_dir / f"yt_topic_{split_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{split_name} CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required_cols = {"text", "target", "attribute"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{split_name} CSV에 필요한 컬럼이 없습니다: {missing}. 현재 컬럼: {list(df.columns)}")

    df = df.loc[:, ["text", "target", "attribute"]]
    df = df.dropna(subset=["text", "target", "attribute"]).copy()
    df["text"] = df["text"].astype(str)
    df["target"] = df["target"].astype(str)
    df["attribute"] = df["attribute"].astype(str)
    return df.reset_index(drop=True)


def multi_hot(labels, num_classes: int) -> np.ndarray:
    vec = np.zeros(num_classes, dtype=np.float32)
    for x in str(labels).split(","):
        x = x.strip()
        if x != "" and x.lower() != "nan":
            vec[int(float(x))] = 1.0
    return vec


class CommentMultiTaskDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.texts = df["text"].astype(str).tolist()
        self.target_labels = [multi_hot(x, NUM_TARGET_CLASSES) for x in df["target"]]
        self.attribute_labels = [multi_hot(x, NUM_ATTRIBUTE_CLASSES) for x in df["attribute"]]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["target_labels"] = torch.tensor(self.target_labels[idx], dtype=torch.float)
        item["attribute_labels"] = torch.tensor(self.attribute_labels[idx], dtype=torch.float)
        return item



# 모델 정의(멀티태스크)
class ElectraMultiTaskClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_target_labels: int,
        num_attribute_labels: int,
        dropout: float = 0.1,
        target_loss_weight: float = 1.0,
        attribute_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        hidden_size = self.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.target_classifier = nn.Linear(hidden_size, num_target_labels)
        self.attribute_classifier = nn.Linear(hidden_size, num_attribute_labels)
        self.target_loss_weight = target_loss_weight
        self.attribute_loss_weight = attribute_loss_weight

    def forward(self, input_ids, attention_mask, token_type_ids=None, target_labels=None, attribute_labels=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = self.dropout(outputs.last_hidden_state[:, 0])

        target_logits = self.target_classifier(pooled_output)
        attribute_logits = self.attribute_classifier(pooled_output)

        loss = target_loss = attribute_loss = None
        if target_labels is not None and attribute_labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            target_loss = loss_fct(target_logits, target_labels)
            attribute_loss = loss_fct(attribute_logits, attribute_labels)
            loss = self.target_loss_weight * target_loss + self.attribute_loss_weight * attribute_loss

        return {
            "loss": loss,
            "target_loss": target_loss,
            "attribute_loss": attribute_loss,
            "target_logits": target_logits,
            "attribute_logits": attribute_logits,
        }



# 평가 및 결과 저장
def apply_threshold_with_fallback(target_probs, attr_probs, target_threshold: float, attribute_threshold: float):
    target_preds = (target_probs >= target_threshold).int()
    attr_preds = (attr_probs >= attribute_threshold).int()

    # 아무 라벨도 선택되지 않으면 fallback 라벨 부여
    target_preds[target_preds.sum(dim=1) == 0, TARGET_LABEL2ID["Others"]] = 1
    attr_preds[attr_preds.sum(dim=1) == 0, ATTRIBUTE_LABEL2ID["None"]] = 1

    return target_preds, attr_preds


def evaluate_multitask(model, data_loader, device, target_threshold: float = 0.48, attribute_threshold: float = 0.33):
    model.eval()
    total_loss = total_target_loss = total_attribute_loss = 0.0
    all_target_preds, all_target_labels, all_target_probs = [], [], []
    all_attr_preds, all_attr_labels, all_attr_probs = [], [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            target_labels = batch["target_labels"].to(device)
            attribute_labels = batch["attribute_labels"].to(device)

            outputs = model(input_ids, attention_mask, token_type_ids, target_labels, attribute_labels)
            total_loss += outputs["loss"].item()
            total_target_loss += outputs["target_loss"].item()
            total_attribute_loss += outputs["attribute_loss"].item()

            target_probs = torch.sigmoid(outputs["target_logits"])
            attr_probs = torch.sigmoid(outputs["attribute_logits"])
            target_preds, attr_preds = apply_threshold_with_fallback(
                target_probs,
                attr_probs,
                target_threshold=target_threshold,
                attribute_threshold=attribute_threshold,
            )

            all_target_probs.extend(target_probs.cpu().numpy())
            all_attr_probs.extend(attr_probs.cpu().numpy())
            all_target_preds.extend(target_preds.cpu().numpy())
            all_target_labels.extend(target_labels.cpu().numpy())
            all_attr_preds.extend(attr_preds.cpu().numpy())
            all_attr_labels.extend(attribute_labels.cpu().numpy())

    all_target_labels = np.array(all_target_labels)
    all_target_preds = np.array(all_target_preds)
    all_target_probs = np.array(all_target_probs)
    all_attr_labels = np.array(all_attr_labels)
    all_attr_preds = np.array(all_attr_preds)
    all_attr_probs = np.array(all_attr_probs)

    target_macro_f1 = f1_score(all_target_labels, all_target_preds, average="macro", zero_division=0)
    attribute_macro_f1 = f1_score(all_attr_labels, all_attr_preds, average="macro", zero_division=0)

    metrics = {
        "loss": total_loss / len(data_loader),
        "target_loss": total_target_loss / len(data_loader),
        "attribute_loss": total_attribute_loss / len(data_loader),
        "target_exact_acc": accuracy_score(all_target_labels, all_target_preds),
        "target_macro_f1": target_macro_f1,
        "target_micro_f1": f1_score(all_target_labels, all_target_preds, average="micro", zero_division=0),
        "target_weighted_f1": f1_score(all_target_labels, all_target_preds, average="weighted", zero_division=0),
        "attribute_exact_acc": accuracy_score(all_attr_labels, all_attr_preds),
        "attribute_macro_f1": attribute_macro_f1,
        "attribute_micro_f1": f1_score(all_attr_labels, all_attr_preds, average="micro", zero_division=0),
        "attribute_weighted_f1": f1_score(all_attr_labels, all_attr_preds, average="weighted", zero_division=0),
        "joint_acc": float(np.mean(
            np.all(all_target_labels == all_target_preds, axis=1)
            & np.all(all_attr_labels == all_attr_preds, axis=1)
        )),
        "mean_macro_f1": (target_macro_f1 + attribute_macro_f1) / 2,
    }

    predictions = {
        "target_true": all_target_labels,
        "target_pred": all_target_preds,
        "target_prob": all_target_probs,
        "attribute_true": all_attr_labels,
        "attribute_pred": all_attr_preds,
        "attribute_prob": all_attr_probs,
    }

    return metrics, predictions


def decode_multi_label(vec, id2label):
    return [id2label[idx] for idx, v in enumerate(vec) if int(v) == 1]


def save_prediction_results(df: pd.DataFrame, preds: dict, save_path: Path) -> pd.DataFrame:
    rows = []
    for i in range(len(df)):
        target_true = preds["target_true"][i]
        target_pred = preds["target_pred"][i]
        attr_true = preds["attribute_true"][i]
        attr_pred = preds["attribute_pred"][i]
        target_probs = preds["target_prob"][i]
        attr_probs = preds["attribute_prob"][i]

        row = {
            "text": df.iloc[i]["text"],
            "target_true": decode_multi_label(target_true, TARGET_ID2LABEL),
            "target_pred": decode_multi_label(target_pred, TARGET_ID2LABEL),
            "attribute_true": decode_multi_label(attr_true, ATTRIBUTE_ID2LABEL),
            "attribute_pred": decode_multi_label(attr_pred, ATTRIBUTE_ID2LABEL),
            "target_correct": bool(np.array_equal(target_true, target_pred)),
            "attribute_correct": bool(np.array_equal(attr_true, attr_pred)),
            "joint_correct": bool(np.array_equal(target_true, target_pred) and np.array_equal(attr_true, attr_pred)),
        }

        for idx, label in TARGET_ID2LABEL.items():
            row[f"target_prob_{label}"] = round(float(target_probs[idx]), 4)
        for idx, label in ATTRIBUTE_ID2LABEL.items():
            row[f"attribute_prob_{label}"] = round(float(attr_probs[idx]), 4)

        rows.append(row)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print("saved predictions:", save_path)
    return result_df


def save_classification_reports(preds: dict, result_dir: Path, prefix: str) -> None:
    target_report = classification_report(
        preds["target_true"],
        preds["target_pred"],
        target_names=[TARGET_ID2LABEL[i] for i in range(NUM_TARGET_CLASSES)],
        output_dict=True,
        zero_division=0,
    )
    attribute_report = classification_report(
        preds["attribute_true"],
        preds["attribute_pred"],
        target_names=[ATTRIBUTE_ID2LABEL[i] for i in range(NUM_ATTRIBUTE_CLASSES)],
        output_dict=True,
        zero_division=0,
    )

    with open(result_dir / f"{prefix}_target_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(target_report, f, ensure_ascii=False, indent=2)

    with open(result_dir / f"{prefix}_attribute_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(attribute_report, f, ensure_ascii=False, indent=2)


def get_score(metrics: dict, selection_metric: str) -> float:
    if selection_metric == "mean_macro_f1":
        return metrics["mean_macro_f1"]
    if selection_metric == "joint_acc":
        return metrics["joint_acc"]
    if selection_metric == "target_macro_f1":
        return metrics["target_macro_f1"]
    if selection_metric == "attribute_macro_f1":
        return metrics["attribute_macro_f1"]
    raise ValueError(f"Invalid selection_metric: {selection_metric}")



# 실행
def run_experiment(args) -> None:
    set_seed(args.seed)

    cache_dir = os.environ.get("HF_HOME")
    output_dir = Path(args.output_dir)
    model_dir = output_dir / "saved_models"
    result_dir = output_dir / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    print("\n\n===== Configuration =====")
    print("EXP_NAME:", args.exp_name)
    print("OUTPUT_DIR:", output_dir)
    print("MODEL_NAME:", args.model_name)
    

    # config 저장
    with open(result_dir / f"{args.exp_name}_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    train_df = load_split_csv(args.data_dir, "train")
    valid_df = load_split_csv(args.data_dir, "valid")
    test_df = load_split_csv(args.data_dir, "test")

    print("\n\n===== Dataset summary =====")
    print("train:", len(train_df))
    print("valid:", len(valid_df))
    print("test :", len(test_df))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n\n===== Model loading =====")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=cache_dir)

    train_dataset = CommentMultiTaskDataset(train_df, tokenizer, max_length=args.max_length)
    valid_dataset = CommentMultiTaskDataset(valid_df, tokenizer, max_length=args.max_length)
    test_dataset = CommentMultiTaskDataset(test_df, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)

    model = ElectraMultiTaskClassifier(
        model_name=args.model_name,
        num_target_labels=NUM_TARGET_CLASSES,
        num_attribute_labels=NUM_ATTRIBUTE_CLASSES,
        dropout=args.dropout,
        target_loss_weight=args.target_loss_weight,
        attribute_loss_weight=args.attribute_loss_weight,
    ).to(device)

    print("Tokenizer loaded.") 
    print("Model loaded.")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    best_score = -1.0
    best_epoch = None
    patience_counter = 0
    best_path = model_dir / f"{args.exp_name}.pt"
    history = []

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0
        loop = tqdm(train_loader, desc=f"{args.exp_name} | Epoch {epoch + 1}/{args.epochs}")

        for batch in loop:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            target_labels = batch["target_labels"].to(device)
            attribute_labels = batch["attribute_labels"].to(device)

            outputs = model(input_ids, attention_mask, token_type_ids, target_labels, attribute_labels)
            loss = outputs["loss"]
            loss.backward()

            if args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        train_loss = total_train_loss / len(train_loader)
        valid_metrics, _ = evaluate_multitask(
            model,
            valid_loader,
            device,
            target_threshold=args.target_threshold,
            attribute_threshold=args.attribute_threshold,
        )
        current_score = get_score(valid_metrics, args.selection_metric)

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **{f"valid_{k}": v for k, v in valid_metrics.items()},
            "selection_score": current_score,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(result_dir / f"{args.exp_name}_history.csv", index=False, encoding="utf-8-sig")

        print(f"\n===== {args.exp_name} | Epoch {epoch + 1} =====")
        print(json.dumps(row, ensure_ascii=False, indent=2))

        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
            print(">> best model saved:", best_path)
        else:
            patience_counter += 1

        if args.early_stopping_patience is not None and patience_counter >= args.early_stopping_patience:
            print(">> Early stopping triggered")
            break

    
    # 최종 평가
    print("\n\n===== Final evaluation =====")
    print("Loading best model:", best_path)
    print("Best epoch:", best_epoch)
    print("Best validation score:", best_score)

    model.load_state_dict(torch.load(best_path, map_location=device))

    print("\nEvaluating validation set...")
    valid_metrics, valid_preds = evaluate_multitask(
        model,
        valid_loader,
        device,
        target_threshold=args.target_threshold,
        attribute_threshold=args.attribute_threshold,
    )
    print("\nValidation metrics:")
    print(json.dumps(valid_metrics, ensure_ascii=False, indent=2))

    print("\nEvaluating test set...")
    test_metrics, test_preds = evaluate_multitask(
        model,
        test_loader,
        device,
        target_threshold=args.target_threshold,
        attribute_threshold=args.attribute_threshold,
    )
    print("\nTest metrics:")
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))

    threshold_info = {
        "target_threshold": args.target_threshold,
        "attribute_threshold": args.attribute_threshold,
    }

    print("\n\n==== Saving results =====")
    save_prediction_results(valid_df, valid_preds, result_dir / f"{args.exp_name}_valid_predictions.csv")
    save_prediction_results(test_df, test_preds, result_dir / f"{args.exp_name}_test_predictions.csv")
    save_classification_reports(valid_preds, result_dir, prefix=f"{args.exp_name}_valid")
    save_classification_reports(test_preds, result_dir, prefix=f"{args.exp_name}_test")

    result_row = {
        "exp_name": args.exp_name,
        "model_name": args.model_name,
        "max_length": args.max_length,
        "lr": args.lr,
        "configured_epochs": args.epochs,
        "actual_epochs": len(history),
        "best_epoch": best_epoch,
        "batch_size": args.batch_size,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "target_loss_weight": args.target_loss_weight,
        "attribute_loss_weight": args.attribute_loss_weight,
        "selection_metric": args.selection_metric,
        "best_valid_score": best_score,
        **{f"valid_{k}": v for k, v in valid_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
        **threshold_info,
        "model_path": str(best_path),
        "history_path": str(result_dir / f"{args.exp_name}_history.csv"),
        "valid_prediction_path": str(result_dir / f"{args.exp_name}_valid_predictions.csv"),
        "test_prediction_path": str(result_dir / f"{args.exp_name}_test_predictions.csv"),
    }

    pd.DataFrame([result_row]).to_csv(result_dir / f"{args.exp_name}_summary.csv", index=False, encoding="utf-8-sig")

    with open(result_dir / f"{args.exp_name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(result_row, f, ensure_ascii=False, indent=2)

    print("\n===== Final summary =====")
    print(json.dumps(result_row, ensure_ascii=False, indent=2))
    print("\nDone.")
    print("saved summary:", result_dir / f"{args.exp_name}_summary.csv")



# 메인
def main():
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
