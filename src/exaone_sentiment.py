# -*- coding: utf-8 -*-

import os
import re
import json
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def str2bool(value):    # do_sample 같은 boolean 인자 처리용
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("boolean 값은 true/false로 입력하세요.")

def parse_args():
    project_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/local_datasets/yt_sentiment_labeled_1605")
    parser.add_argument("--output_dir", type=str, default=str(project_root / "results" / "exaone_sentiment_retrieval_fewshot"))
    parser.add_argument("--model_name", type=str, default="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct")
    # retrieval few-shot에서 사용할 임베딩 모델 선택
    # kosimcse: BM-K/KoSimCSE-roberta, CLS/pooler embedding 사용
    # kcelectra: beomi/KcELECTRA-base-v2022, mean pooling 사용
    parser.add_argument("--embed_backend", type=str, default="kosimcse", choices=["kosimcse", "kcelectra"])
    parser.add_argument("--embed_model_name1", type=str, default="BM-K/KoSimCSE-roberta") # 임베딩 모델1 : KoSimCSE-roberta
    parser.add_argument("--embed_model_name2", type=str, default="beomi/KcELECTRA-base-v2022") # 임베딩 모델2 : KcELECTRA

    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="label")

    parser.add_argument("--eval_split", type=str, default="test", choices=["valid", "test", "both"])
    parser.add_argument("--run_mode", type=str, default="few_shot", choices=["zero_shot", "few_shot", "both"])
    parser.add_argument("--top_k", type=int, default=10)

    # 빠른 디버깅용. 전체 실행이면 None.
    parser.add_argument("--test_sample_size", type=int, default=None)
    parser.add_argument("--valid_sample_size", type=int, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--do_sample", type=str2bool, default=False)
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--embedding_batch_size", type=int, default=32)
    parser.add_argument("--embedding_max_length", type=int, default=128)

    return parser.parse_args()


# 데이터 로드
def load_split_csv(path: Path, text_col: str, label_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"데이터 파일이 없습니다: {path}")

    df = pd.read_csv(path)
    required_cols = {text_col, label_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}에 필요한 컬럼이 없습니다: {missing}. 현재 컬럼: {list(df.columns)}")

    df = df[[text_col, label_col]].dropna().copy()
    df[text_col] = df[text_col].astype(str)
    df[label_col] = df[label_col].astype(int)
    return df.reset_index(drop=True)


# 코드 정상 작동을 확인하기 위한 데이터 일부 샘플링
def sample_df(df: pd.DataFrame, sample_size: int | None, label_col: str, seed: int) -> pd.DataFrame:
    if sample_size is None or sample_size >= len(df):
        return df.copy()

    # 라벨 분포를 최대한 유지해서 샘플링
    return (
        df.groupby(label_col, group_keys=False)
        .apply(lambda x: x.sample(
            n=max(1, round(len(x) / len(df) * sample_size)),
            random_state=seed
        ))
        .sample(frac=1, random_state=seed)
        .head(sample_size)
        .reset_index(drop=True)
    )



# 프롬프트 설정
ID2LABEL = {0: "부정", 1: "중립", 2: "긍정"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

SYSTEM_PROMPT = """
당신은 한국 오디션 프로그램(예: 쇼미더머니, 스트릿 우먼 파이터, 보이즈플래닛)의 유튜브 영상 댓글 감정 분류기입니다.

댓글은 짧고, 비문이 많으며, 밈/드립/팬덤식 표현/반어적 표현이 포함될 수 있습니다.

출력 규칙:
- 반드시 아래 3개 중 하나만 출력
- 다른 문장, 설명, 따옴표 출력 절대 금지

가능한 출력:
부정
중립
긍정
""".strip()

LABEL_GUIDE = """
[부정]
참가자, 무대, 방송, 편집, 결과, 분량, 업로드 등에 대한 비판, 불만, 조롱, 실망, 아쉬움, 부정적 평가가 중심인 댓글.
질문 형태라도 실제 의도가 비판/불만이면 부정으로 분류한다.

[중립]
명확한 칭찬이나 비판 없이 단순 감상, 정보 전달, 단순 질문, 인물 언급, 단순 요청, 피드백 등 감정이 명확히 드러나지 않거나 애매한 댓글.
밈/드립이라도 명확한 칭찬이나 비판이 드러나지 않으면 중립으로 분류한다.

[긍정]
참가자, 무대, 음원, 실력, 외모, 조합 등에 대한 명확한 칭찬, 응원, 호감, 감탄, 긍정적 평가가 중심인 댓글.
단순 팬덤식 언급이 아니라, 긍정 평가가 분명해야 한다.
""".strip()


# 제로샷
def build_zero_shot_prompt(comment: str) -> str:
    return f"""
글의 감정을 부정, 중립, 긍정 중 하나로 분류하세요.
만약 댓글에 여러 감정이 함께 포함된 경우, 작성자가 최종적으로 전달하고자 하는 핵심 감정과 의도를 기준으로 분류하세요.

{LABEL_GUIDE}

반드시 아래 중 하나만 출력하세요:
부정
중립
긍정

댓글:
"{comment}"

정답:
""".strip()


# 퓨샷
def build_few_shot_prompt(comment: str, retrieved_examples: pd.DataFrame, text_col: str, label_col: str) -> str:
    example_text = ""

    for _, row in retrieved_examples.iterrows():
        label_name = ID2LABEL[int(row[label_col])]
        example_text += f"댓글: {row[text_col]}\n"
        example_text += f"정답: {label_name}\n\n"

    return f"""
댓글의 감정을 부정, 중립, 긍정 중 하나로 분류하세요.
만약 댓글에 여러 감정이 함께 포함된 경우, 작성자가 최종적으로 전달하고자 하는 핵심 감정과 의도를 기준으로 분류하세요.

{LABEL_GUIDE}

아래 예시는 분류 기준을 보여주는 참고 예시입니다.
예시와 단어가 비슷하더라도, 반드시 현재 댓글의 핵심 의도를 기준으로 판단하세요.

예시:
{example_text}

분류할 댓글:
"{comment}"

정답:
""".strip()



# 임베딩 / retrieval
def get_active_embed_model_name(args) -> str:
    """--embed_backend 선택값에 따라 실제 사용할 임베딩 모델명을 반환"""
    if args.embed_backend == "kosimcse":
        return args.embed_model_name1
    if args.embed_backend == "kcelectra":
        return args.embed_model_name2
    raise ValueError(f"지원하지 않는 embed_backend입니다: {args.embed_backend}")


def mean_pooling(outputs, attention_mask):
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

    return sum_embeddings / sum_mask


def cls_pooling(outputs):
    """
    SimCSE 계열은 보통 [CLS] representation을 문장 임베딩으로 사용.
    모델에 pooler_output이 있으면 우선 사용하고, 없으면 last_hidden_state[:, 0] 사용.
    """
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output
    return outputs.last_hidden_state[:, 0]


# BM-K/KoSimCSE-roberta로 문장 임베딩 생성
# SimCSE 계열이므로 CLS/pooler embedding 사용
def encode_texts_kosimcse(
    texts,
    embedding_tokenizer,
    embedding_model,
    device,
    batch_size=32,
    max_length=128,
    desc="Encoding with KoSimCSE",
):
    all_embeddings = []
    embedding_model.eval()

    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch_texts = [str(x) for x in texts[i:i + batch_size]]

        inputs = embedding_tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = embedding_model(**inputs, return_dict=True)

        embeddings = cls_pooling(outputs)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)


# KcELECTRA로 문장 임베딩 생성
# 일반 encoder라서 attention mask 기반 mean pooling 사용
def encode_texts_kcelectra(
    texts,
    embedding_tokenizer,
    embedding_model,
    device,
    batch_size=32,
    max_length=128,
    desc="Encoding with KcELECTRA",
):
    all_embeddings = []
    embedding_model.eval()

    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch_texts = [str(x) for x in texts[i:i + batch_size]]

        inputs = embedding_tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = embedding_model(**inputs, return_dict=True)

        embeddings = mean_pooling(outputs, inputs["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)


def encode_texts(
    texts,
    embed_backend,
    embedding_tokenizer,
    embedding_model,
    device,
    batch_size=32,
    max_length=128,
    desc="Encoding",
):
    """선택한 임베딩 백본에 맞는 방식으로 문장 임베딩 생성"""
    if embed_backend == "kosimcse":
        return encode_texts_kosimcse(
            texts=texts,
            embedding_tokenizer=embedding_tokenizer,
            embedding_model=embedding_model,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            desc=desc,
        )

    if embed_backend == "kcelectra":
        return encode_texts_kcelectra(
            texts=texts,
            embedding_tokenizer=embedding_tokenizer,
            embedding_model=embedding_model,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            desc=desc,
        )

    raise ValueError(f"지원하지 않는 embed_backend입니다: {embed_backend}")


def retrieve_similar_examples(
    query_text,
    train_df,
    train_embeddings,
    embedding_tokenizer,
    embedding_model,
    device,
    text_col,
    label_col,
    embed_backend,
    top_k=10,
    embedding_max_length=128,
):
    query_embedding = encode_texts(
        [str(query_text)],
        embed_backend=embed_backend,
        embedding_tokenizer=embedding_tokenizer,
        embedding_model=embedding_model,
        device=device,
        batch_size=1,
        max_length=embedding_max_length,
        desc="Query embedding",
    ).to(device)

    similarities = torch.matmul(query_embedding, train_embeddings.T).squeeze(0)
    top_scores, top_indices = torch.topk(similarities, k=min(top_k, len(train_df)))

    top_indices = top_indices.detach().cpu().numpy()
    top_scores = top_scores.detach().cpu().numpy()

    retrieved = train_df.iloc[top_indices].copy()
    retrieved["similarity"] = top_scores

    return retrieved[[text_col, label_col, "similarity"]]



# 엑사원 생성 및 파싱
def make_messages(user_prompt: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def get_first_model_device(model):
    # device_map="auto"일 때 입력 텐서를 올릴 기본 GPU
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_answer(
    user_prompt: str,
    tokenizer,
    model,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
) -> str:
    messages = make_messages(user_prompt)

    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    # transformers 버전에 따라 Tensor 또는 dict/BatchEncoding으로 반환될 수 있어 둘 다 처리
    if isinstance(encoded, torch.Tensor):
        input_ids = encoded
    else:
        input_ids = encoded["input_ids"]

    input_ids = input_ids.to(get_first_model_device(model))

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def parse_label(output_text: str):
    text = str(output_text).strip()

    for label in ["부정", "중립", "긍정"]:
        if label in text:
            return label

    digit_map = {"0": "부정", "1": "중립", "2": "긍정"}
    for digit, label in digit_map.items():
        if re.search(rf"(^|\D){digit}(\D|$)", text):
            return label

    return "파싱실패"



# 평가
def run_llm_classification(
    eval_df: pd.DataFrame,
    mode: str,
    args,
    tokenizer,
    model,
    train_df=None,
    train_embeddings=None,
    embedding_tokenizer=None,
    embedding_model=None,
    embedding_device=None,
) -> pd.DataFrame:
    assert mode in ["zero_shot", "few_shot"]

    rows = []
    iterator = tqdm(eval_df.reset_index(drop=True).iterrows(), total=len(eval_df), desc=mode)

    for idx, row in iterator:
        comment = row[args.text_col]
        true_id = int(row[args.label_col])
        true_label = ID2LABEL[true_id]

        retrieved_text = ""

        if mode == "zero_shot":
            prompt = build_zero_shot_prompt(comment)
        else:
            retrieved_examples = retrieve_similar_examples(
                query_text=comment,
                train_df=train_df,
                train_embeddings=train_embeddings,
                embedding_tokenizer=embedding_tokenizer,
                embedding_model=embedding_model,
                device=embedding_device,
                text_col=args.text_col,
                label_col=args.label_col,
                embed_backend=args.embed_backend,
                top_k=args.top_k,
                embedding_max_length=args.embedding_max_length,
            )
            prompt = build_few_shot_prompt(
                comment=comment,
                retrieved_examples=retrieved_examples,
                text_col=args.text_col,
                label_col=args.label_col,
            )
            retrieved_text = json.dumps(
                [
                    {
                        "text": str(r[args.text_col]),
                        "label": ID2LABEL[int(r[args.label_col])],
                        "similarity": float(r["similarity"]),
                    }
                    for _, r in retrieved_examples.iterrows()
                ],
                ensure_ascii=False,
            )

        try:
            raw_output = generate_answer(
                prompt,
                tokenizer=tokenizer,
                model=model,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
            )
            pred_label = parse_label(raw_output)
        except Exception as e:
            raw_output = f"ERROR: {repr(e)}"
            pred_label = "파싱실패"

        pred_id = LABEL2ID.get(pred_label, -1)

        rows.append({
            "index": idx,
            "text": comment,
            "true_id": true_id,
            "true_label": true_label,
            "pred_id": pred_id,
            "pred_label": pred_label,
            "raw_output": raw_output,
            "retrieved_examples": retrieved_text,
            "mode": mode,
            "top_k": args.top_k if mode == "few_shot" else None,
            "model_name": args.model_name,
            "embed_backend": args.embed_backend if mode == "few_shot" else None,
            "embed_model_name": args.active_embed_model_name if mode == "few_shot" else None,
        })

    return pd.DataFrame(rows)


def evaluate_result(result_df: pd.DataFrame, mode_name: str, output_dir: Path):
    valid_pred_df = result_df[result_df["pred_id"] != -1].copy()
    failed_df = result_df[result_df["pred_id"] == -1].copy()

    y_true = valid_pred_df["true_id"].to_numpy()
    y_pred = valid_pred_df["pred_id"].to_numpy()

    metrics = {
        "mode": mode_name,
        "n_total": len(result_df),
        "n_valid_pred": len(valid_pred_df),
        "n_parse_failed": len(failed_df),
        "accuracy": accuracy_score(y_true, y_pred) if len(valid_pred_df) else np.nan,
        "macro_f1": f1_score(y_true, y_pred, average="macro") if len(valid_pred_df) else np.nan,
        "weighted_f1": f1_score(y_true, y_pred, average="weighted") if len(valid_pred_df) else np.nan,
    }

    print("\n====", mode_name, "====")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    report_dict = {}
    cm = None

    if len(valid_pred_df):
        print("\nClassification report:")
        print(classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=[ID2LABEL[i] for i in [0, 1, 2]],
            digits=4,
            zero_division=0,
        ))

        report_dict = classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=[ID2LABEL[i] for i in [0, 1, 2]],
            digits=4,
            zero_division=0,
            output_dict=True,
        )

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        print("Confusion matrix:")
        print(cm)

    with open(output_dir / f"{mode_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    if report_dict:
        with open(output_dir / f"{mode_name}_classification_report.json", "w", encoding="utf-8") as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2)

    if cm is not None:
        pd.DataFrame(
            cm,
            index=[f"true_{ID2LABEL[i]}" for i in [0, 1, 2]],
            columns=[f"pred_{ID2LABEL[i]}" for i in [0, 1, 2]],
        ).to_csv(output_dir / f"{mode_name}_confusion_matrix.csv", encoding="utf-8-sig")

    return metrics



# 메인
def main():
    args = parse_args()
    args.active_embed_model_name = get_active_embed_model_name(args)
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n\n===== Configuration =====")
    print("OUTPUT_DIR:", output_dir)
    print("MODEL_NAME:", args.model_name)
    print("EMBED_BACKEND:", args.embed_backend)
    print("EMBED_MODEL_NAME:", args.active_embed_model_name)
    print("RUN_MODE:", args.run_mode)
    print("EVAL_SPLIT:", args.eval_split)
    print("TOP_K:", args.top_k)

    train_df = load_split_csv(data_dir / "yt_train.csv", args.text_col, args.label_col)
    valid_df = load_split_csv(data_dir / "yt_valid.csv", args.text_col, args.label_col)
    test_df = load_split_csv(data_dir / "yt_test.csv", args.text_col, args.label_col)

    valid_eval_df = sample_df(valid_df, args.valid_sample_size, args.label_col, args.seed)
    test_eval_df = sample_df(test_df, args.test_sample_size, args.label_col, args.seed)

    print("\n\n===== Dataset summary =====")
    print("train:", len(train_df))
    print("valid:", len(valid_df), "valid_eval:", len(valid_eval_df))
    print("test:", len(test_df), "test_eval:", len(test_eval_df))
    print("\nTrain label distribution:")
    print(train_df[args.label_col].value_counts().sort_index())
    print("\nValid label distribution:")
    print(valid_eval_df[args.label_col].value_counts().sort_index())
    print("\nTest label distribution:")
    print(test_eval_df[args.label_col].value_counts().sort_index())

    # EXAONE 로드
    print("\n\n===== Model loading =====")
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        torch_dtype = torch.float32

    print("\ntorch_dtype:", torch_dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    model.eval()
    print("LLM loaded.")

    # few-shot이면 embedding backbone 로드
    embedding_tokenizer = None
    embedding_model = None
    embedding_device = None
    train_embeddings = None

    if args.run_mode in ["few_shot", "both"]:
        embedding_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embedding_tokenizer = AutoTokenizer.from_pretrained(
            args.active_embed_model_name,
            cache_dir=cache_dir,
        )
        embedding_model = AutoModel.from_pretrained(
            args.active_embed_model_name,
            cache_dir=cache_dir,
        ).to(embedding_device)
        embedding_model.eval()
        print("Embedding model loaded.")

        train_texts = train_df[args.text_col].astype(str).tolist()
        train_embeddings = encode_texts(
            train_texts,
            embed_backend=args.embed_backend,
            embedding_tokenizer=embedding_tokenizer,
            embedding_model=embedding_model,
            device=embedding_device,
            batch_size=args.embedding_batch_size,
            max_length=args.embedding_max_length,
            desc=f"Encoding train set with {args.embed_backend}",
        ).to(embedding_device)

        print("train_embeddings:", train_embeddings.shape)

    # 샘플 테스트
    print("\n\n===== Sample inference =====")
    sample_comment = "이 무대가 1위라는 게 이해 안 되면 개추"
    sample_raw = generate_answer(
        build_zero_shot_prompt(sample_comment),
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
    )
    print("Comment:", sample_comment)
    print("Raw output:", sample_raw)
    print("Parsed label:", parse_label(sample_raw))

    print(f"\n\n===== Evaluation =====")
    eval_targets = []
    if args.eval_split in ["valid", "both"]:
        eval_targets.append(("valid", valid_eval_df))
    if args.eval_split in ["test", "both"]:
        eval_targets.append(("test", test_eval_df))

    modes = []
    if args.run_mode in ["zero_shot", "both"]:
        modes.append("zero_shot")
    if args.run_mode in ["few_shot", "both"]:
        modes.append("few_shot")

    all_metrics = []

    for split_name, eval_df in eval_targets:
        for mode in modes:
            mode_name = f"{split_name}_{mode}"
            if mode == "few_shot":
                mode_name += f"_{args.embed_backend}_top{args.top_k}"

            result_df = run_llm_classification(
                eval_df=eval_df,
                mode=mode,
                args=args,
                tokenizer=tokenizer,
                model=model,
                train_df=train_df,
                train_embeddings=train_embeddings,
                embedding_tokenizer=embedding_tokenizer,
                embedding_model=embedding_model,
                embedding_device=embedding_device,
            )

            pred_path = output_dir / f"{mode_name}_predictions.csv"
            result_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
            print("saved predictions:", pred_path)

            metrics = evaluate_result(result_df, mode_name=mode_name, output_dir=output_dir)
            all_metrics.append(metrics)

    with open(output_dir / "all_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print("saved all metrics:", output_dir / "all_metrics.json")


if __name__ == "__main__":
    main()
