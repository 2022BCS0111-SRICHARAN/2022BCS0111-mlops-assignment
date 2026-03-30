# End-to-End MLOps Pipeline Assignment

**Name:** Sricharan  
**Roll No:** 2022BCS0111  
**GitHub Repository:** `<rollno>-mlops`  
**Docker Image:** `<dockerhub-username>/sricharan-mlops`  
**MLflow Experiment Name:** `sricharan_experiment`

---

## 1. Problem Statement

- **Type:** Regression
- **Objective:** Predict the quality of red wine based on physicochemical properties.
- **Target Variable:** `quality` (integer score between 0 and 10)

## 2. Dataset Description

- **Source:** UCI Machine Learning Repository — Wine Quality Dataset
- **File:** `winequality-red.csv`
- **Samples:** 1,599
- **Features:** 11 numeric input features
  - `fixed_acidity`, `volatile_acidity`, `citric_acid`, `residual_sugar`, `chlorides`, `free_sulfur_dioxide`, `total_sulfur_dioxide`, `density`, `pH`, `sulphates`, `alcohol`
- **Target:** `quality`
- **Dataset Size:** ~84 KB
- **Preprocessing Steps:**
  - Standard scaling of features
  - Train-test split (70-30, random_state=42)
  - Dataset v2: Outlier removal (IQR method) for improved version

## 3. Repository Structure

```
assignment/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .dvc/
│   └── config                         # DVC remote storage config
├── dvc.yaml                           # DVC pipeline stages
├── dataset/
│   ├── winequality-red.csv            # Dataset v1 (original)
│   └── winequality-red.csv.dvc        # DVC tracking
├── src/
│   └── train.py                       # Training script with MLflow
├── app/
│   └── main.py                        # FastAPI endpoints
├── Dockerfile                         # Container build
├── .github/
│   └── workflows/
│       └── mlops-pipeline.yml         # CI/CD pipeline
├── tests/
│   └── test_inference.py              # Inference validation
├── models/
│   └── model.pkl                      # Trained model
├── output_metrics.json                # Results from all 5 runs
└── REPORT.md                          # Analysis & comparison
```

## 4. How to Run

### Setup
```bash
pip install -r requirements.txt
```

### Train Model (with MLflow)
```bash
# Run 1: Lasso baseline (all features)
python src/train.py --run_name run1 --model_type lasso --alpha 0.1 --dataset_version v1

# Run 2: Ridge (all features)
python src/train.py --run_name run2 --model_type ridge --alpha 1.0 --dataset_version v1

# Run 3: Lasso with different alpha
python src/train.py --run_name run3 --model_type lasso --alpha 0.01 --dataset_version v1

# Run 4: RandomForest with feature selection (dataset v2)
python src/train.py --run_name run4 --model_type random_forest --n_estimators 100 --max_depth 10 --dataset_version v2 --features selected

# Run 5: GradientBoosting with feature selection (dataset v2)
python src/train.py --run_name run5 --model_type gradient_boosting --n_estimators 100 --max_depth 5 --dataset_version v2 --features selected
```

### View MLflow UI
```bash
mlflow ui --port 5000
```

### Start API Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker
```bash
docker build -t sricharan-mlops .
docker tag sricharan-mlops <dockerhub-username>/sricharan-mlops:latest
docker push <dockerhub-username>/sricharan-mlops:latest
```

### Run Tests
```bash
pytest tests/
```

### DVC
```bash
dvc init
dvc remote add -d myremote s3://<bucket-name>/dvc-store
dvc add dataset/winequality-red.csv
dvc push
```
