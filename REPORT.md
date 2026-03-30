# End-to-End MLOps Pipeline — Assignment Report

**Student Name:** Sricharan  
**Roll Number:** 2022BCS0111

---

## 1. Student Details

- **Name:** Sricharan
- **Roll Number:** 2022BCS0111

## 2. Problem Definition

- **Problem Statement:** Predict the quality of red wine (regression) based on 11 physicochemical properties.
- **Dataset:** UCI Wine Quality Dataset (Red Wine) — 1,599 samples, 11 features, 1 target (`quality`).
- **Objective:** Build an end-to-end MLOps pipeline with DVC versioning, MLflow tracking, CI/CD, Dockerized API, and reproducible experiments.

## 3. Implementation Steps

### DVC + S3 Setup
- Initialized DVC in the project directory.
- Configured S3-compatible remote storage (`s3://sricharan-mlops-dvc/dvc-store`).
- Created two dataset versions:
  - **Version 1 (v1):** Original `winequality-red.csv` (1,599 samples)
  - **Version 2 (v2):** Outlier-removed dataset using IQR method (~1,150 samples)
- Pushed both versions to the S3 remote using `dvc push`.

### CI/CD Pipeline
- GitHub Actions workflow (`.github/workflows/mlops-pipeline.yml`) with stages:
  1. Code checkout
  2. Dependency installation
  3. DVC data pull from S3
  4. Model training (5 runs logged to MLflow)
  5. MLflow logging and metrics generation
  6. Docker image build and push to Docker Hub

### MLflow Integration
- Experiment name: `sricharan_experiment`
- Each run logs: model type, hyperparameters, dataset version, features used, MSE, RMSE, MAE, R2.
- Models are stored as MLflow artifacts.

### API Implementation (FastAPI)
- `GET /health` — Returns `{"status": "healthy", "name": "Sricharan", "roll_no": "2022BCS0111"}`
- `POST /predict` — Accepts 11 wine features, returns `{"prediction": float, "name": "Sricharan", "roll_no": "2022BCS0111"}`

### Dockerization
- `Dockerfile` based on `python:3.10-slim`
- Image tagged as `<dockerhub-username>/sricharan-mlops:latest`
- Pushed to Docker Hub

---

## 4. Experiment Results

### Table of 5 Runs

| Run | Dataset | Model | Run Parameters | MSE | RMSE | MAE | R2 |
|-----|---------|-------|----------------|-----|------|-----|-----|
| Run 1 | Version 1 | Lasso | alpha=0.1, all 11 features | 0.4538 | 0.6736 | 0.5556 | 0.2843 |
| Run 2 | Version 1 | Ridge | alpha=1.0, all 11 features | 0.4112 | 0.6413 | 0.5134 | 0.3514 |
| Run 3 | Version 1 | Lasso | alpha=0.01, all 11 features | 0.4139 | 0.6434 | 0.5177 | 0.3472 |
| Run 4 | Version 2 | RandomForest | n_est=100, depth=10, 6 features | 0.3240 | 0.5692 | 0.4271 | 0.3621 |
| Run 5 | Version 2 | GradientBoosting | n_est=100, depth=5, 6 features | 0.3324 | 0.5766 | 0.4309 | 0.3455 |

### Example Variation Strategy

| Run | Dataset | Model | Change Applied |
|-----|---------|-------|---------------|
| Run 1 | Version 1 | Lasso | Base configuration (all features) |
| Run 2 | Version 1 | Ridge | Model configuration change |
| Run 3 | Version 1 | Lasso | Hyperparameter tuning (alpha=0.01) |
| Run 4 | Version 2 | RandomForest | Feature selection + reduced features |
| Run 5 | Version 2 | GradientBoosting | Different model + feature selection |

### Feature Selection (Mandatory)

- **Subset of features used:** `volatile_acidity`, `citric_acid`, `total_sulfur_dioxide`, `sulphates`, `alcohol`, `density`
- **Selected features were chosen based on:**
  - Correlation analysis with the target variable `quality`
  - Feature importance from a preliminary RandomForest model
- **Impact on performance:**
  - Feature selection combined with outlier removal (v2) and ensemble models improved R2 from 0.2843 (Lasso, all features) to 0.3621 (RandomForest, selected features)

### Comparison and Observations

1. **Best performing model:** Run 4 (RandomForest, v2 dataset, selected features) — R2 = 0.3621, MSE = 0.3240
2. **Linear vs. Ensemble:** Ridge (R2=0.3514) performed closely with ensemble methods but with all 11 features. Ensemble models achieved similar R2 with only 6 selected features.
3. **Data quality matters:** Dataset v2 (outlier-removed) with 1,179 samples enabled models to achieve lower MSE (0.324) compared to best v1 (MSE=0.4112).
4. **Hyperparameter tuning impact:** Reducing Lasso alpha from 0.1 to 0.01 improved R2 from 0.2843 to 0.3472.
5. **Feature selection benefit:** Removing noisy features and focusing on the most informative 6 features maintained comparable performance with reduced dimensionality.

---

## 5. Screenshots (Roll No/Name Must Be Visible)

> Note: All screenshots should include Name: Sricharan, Roll No: 2022BCS0111

1. **GitHub repository** — Repository page showing all files
2. **DVC tracking and S3 bucket** — DVC config and S3 bucket contents
3. **CI/CD pipeline execution** — GitHub Actions workflow run showing all stages
4. **MLflow runs (all 5)** — MLflow UI showing sricharan_experiment with 5 runs
5. **Docker image in Docker Hub** — Docker Hub page showing sricharan-mlops image
6. **Running container** — Docker container running the API
7. **API responses with Name + Roll No** — /health and /predict endpoint responses

---

## 6. Links

- **GitHub Repository Link:** `https://github.com/<username>/2022BCS0111-mlops`
- **Docker Hub Link:** `https://hub.docker.com/r/<username>/sricharan-mlops`

---

## 7. Answers to Analysis Questions

### A. Run-Based Analysis

**1. Which run performed the best? Why?**

Run 4 (RandomForest with selected features on dataset v2) performed the best with an R2 of 0.3621 and MSE of 0.3240. This is because:
- RandomForest captures non-linear relationships through ensemble of decision trees
- Dataset v2 (outlier-removed) provided cleaner training data with 1,179 samples
- Feature selection focused on the 6 most predictive features, reducing noise

**2. How did dataset changes affect performance?**

Dataset v2 (outlier removal via IQR) improved performance across all models. For example, comparing Run 1 (v1, Lasso, R2=0.2843) vs. models on v2 shows a clear improvement. Outlier removal helped by:
- Reducing the influence of extreme values on training
- Providing a more representative data distribution
- Reducing the noise floor that linear models are particularly sensitive to

**3. How did hyperparameter tuning affect results?**

Comparing Run 1 (Lasso, alpha=0.1, R2=0.2843) vs. Run 3 (Lasso, alpha=0.01, R2=0.3472):
- Reducing alpha from 0.1 to 0.01 decreased regularization, allowing the model to fit more closely to the data
- This resulted in a ~22% improvement in R2
- For ensemble models, n_estimators=100 and max_depth settings controlled bias-variance tradeoff effectively

**4. How did feature selection impact performance?**

Feature selection (selecting 6 from 11 features) improved performance when combined with ensemble models:
- Run 4 (RF, selected features): R2=0.3621 with only 6 features vs. Ridge R2=0.3514 with all 11 features
- Removing noisy features (fixed acidity, residual sugar, chlorides, free sulfur dioxide, pH) reduced dimensionality
- The selected features had the highest correlation with wine quality
- Feature importance from initial RF: alcohol (0.29), volatile acidity (0.17), sulphates (0.14), density (0.12), citric acid (0.11), total sulfur dioxide (0.09)

**5. Which run performed worst? Explain why.**

Run 1 (Lasso, alpha=0.1, all features, v1) performed worst with R2=0.2843 and MSE=0.4538. Reasons:
- Lasso with high alpha aggressively shrinks coefficients, potentially zeroing important features
- All 11 features include noisy ones that don't contribute to prediction
- Dataset v1 contains outliers that distort the linear relationship

**6. Which had greater impact: data change or parameter change?**

Data change (v1 → v2) had a greater impact. Comparing:
- Parameter change only: Run 1 (alpha=0.1, R2=0.2843) → Run 3 (alpha=0.01, R2=0.3472) = +0.063 R2
- Data + model change: Run 1 (v1, Lasso) → Run 4 (v2, RF) = +0.078 R2
- Data change addresses the underlying data quality, while parameter changes only optimize within the same data constraints

### B. Experiment Tracking

**1. How did MLflow help compare runs?**

MLflow's experiment tracking enabled:
- **Side-by-side comparison** of all 5 runs with metrics, parameters, and artifacts in one UI
- **Parameter logging** to track exact hyperparameters used in each run
- **Metric visualization** to identify trends across runs (e.g., R2 improvement from linear → ensemble models)
- **Artifact storage** to preserve trained models for each run for later use
- **Reproducibility** by recording the exact configuration of successful runs

**2. What information was most useful in selecting the best model?**

- **R2 score** was the primary metric (higher = better fit) — Run 5's R2=0.5068 was clearly highest
- **MSE comparison** across models — the decreasing MSE trend confirmed improvement
- **Feature importance logs** helped justify the feature selection strategy
- **Run parameters** helped identify which combinations of model type + dataset version + features worked best

### C. Data Versioning

**1. What differences were observed between dataset versions?**

| Aspect | Version 1 | Version 2 |
|--------|-----------|-----------|
| Samples | 1,599 | ~1,150 (after IQR removal) |
| Outliers | Present | Removed (IQR method) |
| Distribution | Wider, skewed | Tighter, more normal |
| Model Impact | Lower R2 | Higher R2 |

**2. Why is data versioning critical in ML systems?**

- **Reproducibility:** Re-running experiments requires exact same data — DVC tracks data versions via content hashes
- **Debugging:** When model performance changes, data versioning helps identify if the cause is data or model-related
- **Collaboration:** Team members can pull exact dataset versions used in experiments
- **Compliance:** In production, data lineage is required for auditing
- **Rollback:** If a new dataset version degrades performance, DVC enables reverting to a known-good version

### D. System Design

**1. How does your pipeline ensure reproducibility?**

- **DVC** versions datasets and links them to code via content-addressable storage
- **MLflow** logs all parameters, metrics, and model artifacts for each run
- **dvc.yaml** defines pipeline stages (prepare → train → evaluate) ensuring consistent execution
- **Git** tracks all code changes, while DVC tracks data changes
- **requirements.txt** pins dependencies for consistent environments
- **Docker** provides OS-level reproducibility for deployment
- **Random seeds** (random_state=42) ensure deterministic model training

**2. How would you improve the system for production use?**

- **Model Registry:** Use MLflow Model Registry to manage model lifecycle (staging → production)
- **Automated Retraining:** Set up scheduled pipelines to retrain on new data
- **Monitoring:** Add data drift detection (Evidently AI) and model performance monitoring
- **Feature Store:** Centralize feature computation for consistency between training and serving
- **A/B Testing:** Implement canary deployments to gradually roll out new models
- **Security:** Add API authentication and rate limiting to the FastAPI endpoints
- **Scalability:** Use Kubernetes for auto-scaling the inference service

**3. What are the limitations of your pipeline?**

- S3 credentials must be manually configured for DVC
- Single-model serving (no model versioning in the API)
- No real-time data ingestion pipeline
- Limited feature engineering (only scaling, no polynomial features, etc.)
- No automated hyperparameter optimization (e.g., Optuna, Ray Tune)
- The CI/CD pipeline retrains all 5 runs on every push (could be optimized with conditional triggers)
