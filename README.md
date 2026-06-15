# 유튜브 댓글 기반 사용자 감정 및 반응 유형 분류

본 저장소는 유튜브 댓글 데이터를 활용하여 사용자 감정과 반응 유형을 분류한 졸업 프로젝트의 구현 코드를 포함합니다.

본 프로젝트는 다음 두 가지 분류 과제로 구성됩니다.

1. **감정 분류**

   * 댓글을 부정, 중립, 긍정으로 분류합니다.
   * EXAONE 기반 retrieval-based few-shot 방식을 사용합니다.
   * 유사 예시 검색을 위한 임베딩 모델로 KoSimCSE와 KcELECTRA를 지원합니다.

2. **반응 유형 분류**

   * 댓글의 반응 대상인 Target과 반응 속성인 Attribute를 분류합니다.
   * KcELECTRA 기반 멀티태스크·멀티라벨 분류 구조를 사용합니다.

## 저장소 구조

```text
yt_analysis/
├── data/
│   ├── yt_sentiment_labeled_1605.tar
│   └── yt_topic_labeled_2189.tar
├── src/
│   ├── exaone_sentiment.py
│   └── train_topic_multitask.py
├── scripts/
│   ├── run_exaone_sentiment.sh
│   └── run_topic_multitask.sh
├── .gitignore
└── README.md
```

## 분류 과제

### 1. 감정 분류

각 댓글은 다음 세 가지 감정 범주 중 하나로 분류됩니다.

| ID | 라벨 |
| -: | -- |
|  0 | 부정 |
|  1 | 중립 |
|  2 | 긍정 |

EXAONE 모델을 사용하여 감정을 분류하며, 훈련 데이터에서 입력 댓글과 유사한 예시를 검색하여 few-shot 프롬프트에 활용합니다.

지원하는 임베딩 모델은 다음과 같습니다.

* `BM-K/KoSimCSE-roberta`
* `beomi/KcELECTRA-base-v2022`

### 2. 반응 유형 분류

각 댓글은 Target과 Attribute의 두 라벨 체계로 분류됩니다.

#### Target

| ID | 라벨        |
| -: | --------- |
|  0 | Performer |
|  1 | Content   |
|  2 | System    |
|  3 | Others    |

#### Attribute

| ID | 라벨             |
| -: | -------------- |
|  0 | Skill          |
|  1 | Visual         |
|  2 | Character      |
|  3 | Production     |
|  4 | Survival       |
|  5 | Interpretation |
|  6 | None           |

하나의 댓글은 여러 개의 Target 또는 Attribute 라벨을 동시에 가질 수 있습니다.

## 데이터 형식

### 감정 분류 데이터

감정 분류 데이터는 다음 컬럼을 포함해야 합니다.

```text
text,label
```

예시:

```csv
text,label
"무대 진짜 잘했다",2
"업로드 언제 올라오나요?",1
"편집이 너무 아쉽다",0
```

필요한 파일은 다음과 같습니다.

```text
yt_train.csv
yt_valid.csv
yt_test.csv
```

### 반응 유형 분류 데이터

반응 유형 분류 데이터는 다음 컬럼을 포함해야 합니다.

```text
text,target,attribute
```

멀티라벨 데이터의 경우 라벨 ID를 쉼표로 구분합니다.

예시:

```csv
text,target,attribute
"실력도 좋고 무대 연출도 멋있다","0,1","0,3"
```

필요한 파일은 다음과 같습니다.

```text
yt_topic_train.csv
yt_topic_valid.csv
yt_topic_test.csv
```

공개된 데이터에는 댓글 텍스트와 수작업으로 부여한 라벨만 포함되어 있습니다.
(사용자명, 채널 ID, 댓글 URL, 작성 시각 등의 사용자 메타데이터는 포함X)

## 실행 환경

본 코드는 Python과 PyTorch를 기반으로 구현되었습니다.

주요 라이브러리는 다음과 같습니다.

```text
torch
transformers
pandas
numpy
scikit-learn
tqdm
```

## 실험 실행 방법

제공된 shell script는 SLURM 기반 GPU 서버 환경을 기준으로 작성되었습니다.

### 감정 분류

```bash
sbatch scripts/run_exaone_sentiment.sh
```

Python 파일을 직접 실행하려면 다음과 같이 사용할 수 있습니다.

```bash
python src/exaone_sentiment.py \
  --data_dir /path/to/sentiment_dataset \
  --output_dir results/exaone_sentiment \
  --embed_backend kcelectra \
  --run_mode few_shot \
  --eval_split test \
  --top_k 10
```

### 반응 유형 분류

```bash
sbatch scripts/run_topic_multitask.sh
```

Python 파일을 직접 실행하려면 다음과 같이 사용할 수 있습니다.

```bash
python src/train_topic_multitask.py \
  --data_dir /path/to/topic_dataset \
  --output_dir results/topic_multitask \
  --model_name beomi/KcELECTRA-base-v2022 \
  --exp_name topic_multitask_kcelectra \
  --epochs 10 \
  --batch_size 16 \
  --target_threshold 0.48 \
  --attribute_threshold 0.33
```

## 출력 결과

실험 결과는 지정한 출력 디렉터리에 저장됩니다.

생성되는 주요 파일은 다음과 같습니다.

```text
predictions.csv
metrics.json
classification_report.json
confusion_matrix.csv
history.csv
summary.csv
saved_models/
```

## 사용 모델

본 프로젝트에서는 다음 사전학습 모델을 사용합니다.

* `LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct`
* `BM-K/KoSimCSE-roberta`
* `beomi/KcELECTRA-base-v2022`
