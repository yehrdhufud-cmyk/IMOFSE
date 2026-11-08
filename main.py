import time
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import loguniform, randint

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV,
    cross_validate,
    learning_curve
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

from sklearn.inspection import permutation_importance


# ==========================================================
# 1. CONFIGURATION
# ==========================================================

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent

RANDOM_STATE = 42
N_SPLITS = 5
N_ITER = 30

sns.set_theme(style="darkgrid")

COLORS = {
    "background": "#080B14",
    "panel": "#111827",
    "cyan": "#00E5FF",
    "blue": "#3B82F6",
    "purple": "#A855F7",
    "pink": "#EC4899",
    "green": "#22C55E",
    "orange": "#F59E0B",
    "red": "#EF4444",
    "white": "#F8FAFC",
    "muted": "#94A3B8",
    "grid": "#293548"
}

plt.rcParams.update({
    "figure.facecolor": COLORS["background"],
    "axes.facecolor": COLORS["panel"],
    "axes.edgecolor": COLORS["grid"],
    "axes.labelcolor": COLORS["white"],
    "axes.titlecolor": COLORS["white"],
    "xtick.color": COLORS["muted"],
    "ytick.color": COLORS["muted"],
    "text.color": COLORS["white"],
    "grid.color": COLORS["grid"],
    "grid.alpha": 0.45,
    "font.size": 11,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "legend.facecolor": COLORS["panel"],
    "legend.edgecolor": COLORS["grid"],
    "savefig.facecolor": COLORS["background"]
})


def save_chart(filename):
    plt.tight_layout()

    plt.savefig(
        BASE_DIR / filename,
        dpi=300,
        bbox_inches="tight",
        facecolor=COLORS["background"]
    )

    plt.show()
    plt.close()


def style_axis(ax, title, xlabel=None, ylabel=None):

    ax.set_title(title, pad=18, fontweight="bold")

    if xlabel:
        ax.set_xlabel(xlabel)

    if ylabel:
        ax.set_ylabel(ylabel)

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.7,
        alpha=0.35
    )

    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])


print("=" * 70)
print("IMOFSE - ML OPTIMIZATION ENGINE")
print("=" * 70)

experiment_start = time.time()


# ==========================================================
# 2. DATASET
# ==========================================================

dataset = load_breast_cancer()

X = pd.DataFrame(
    dataset.data,
    columns=dataset.feature_names
)

y = pd.Series(dataset.target)

feature_names = X.columns.tolist()

print("\n[DATASET INFORMATION]")

print("Samples:", X.shape[0])
print("Features:", X.shape[1])
print("Classes:", dataset.target_names)
print("Missing values:", X.isnull().sum().sum())
print("Duplicate rows:", X.duplicated().sum())


# ==========================================================
# 3. TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)


# ==========================================================
# 4. CROSS VALIDATION
# ==========================================================

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ==========================================================
# 5. PIPELINE
# ==========================================================

def create_pipeline(model):

    return Pipeline([
        ("scaler", StandardScaler()),

        ("selector", SelectKBest(
            score_func=f_classif
        )),

        ("classifier", model)
    ])


# ==========================================================
# 6. MODEL CONFIGURATION
# ==========================================================

model_configs = {

    "LogisticRegression": {

        "model": LogisticRegression(
            max_iter=5000,
            random_state=RANDOM_STATE
        ),

        "grid": {
            "selector__k": [5, 10, 15, 20, "all"],
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__solver": ["lbfgs", "liblinear"]
        },

        "random": {
            "selector__k": [5, 10, 15, 20, "all"],
            "classifier__C": loguniform(0.001, 1000),
            "classifier__solver": ["lbfgs", "liblinear"]
        }
    },

    "RandomForest": {

        "model": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "grid": {
            "selector__k": [5, 10, 15, 20, "all"],
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [None, 5, 10, 20],
            "classifier__min_samples_split": [2, 5],
            "classifier__max_features": ["sqrt", 0.5]
        },

        "random": {
            "selector__k": [5, 10, 15, 20, "all"],
            "classifier__n_estimators": randint(100, 501),
            "classifier__max_depth": [None, 5, 10, 20, 30],
            "classifier__min_samples_split": randint(2, 11),
            "classifier__max_features": ["sqrt", 0.5, None]
        }
    },

    "SVM": {

        "model": SVC(
            probability=True,
            random_state=RANDOM_STATE
        ),

        "grid": {
            "selector__k": [5, 10, 15, 20, "all"],
            "classifier__C": [0.1, 1, 10, 100],
            "classifier__kernel": ["linear", "rbf"],
            "classifier__gamma": ["scale", "auto"]
        },

        "random": {
            "selector__k": [5, 10, 15, 20, "all"],
            "classifier__C": loguniform(0.001, 1000),
            "classifier__kernel": ["linear", "rbf"],
            "classifier__gamma": loguniform(0.0001, 1)
        }
    }
}


# ==========================================================
# 7. HYPERPARAMETER OPTIMIZATION
# ==========================================================

optimization_results = []
best_estimators = {}

for model_name, config in model_configs.items():

    print("\n" + "=" * 70)
    print("OPTIMIZING:", model_name)
    print("=" * 70)

    pipeline = create_pipeline(config["model"])

    searches = {

        "GridSearch": GridSearchCV(
            estimator=pipeline,
            param_grid=config["grid"],
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=0
        ),

        "RandomizedSearch": RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=config["random"],
            n_iter=N_ITER,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            refit=True,
            verbose=0
        )
    }

    for search_name, search in searches.items():

        print("\nRunning:", search_name)

        start = time.time()

        search.fit(X_train, y_train)

        elapsed = time.time() - start

        best_model = search.best_estimator_

        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        metrics = {

            "Model": model_name,
            "Search": search_name,

            "CV_ROC_AUC": search.best_score_,

            "Test_Accuracy": accuracy_score(y_test, y_pred),

            "Test_Precision": precision_score(y_test, y_pred),

            "Test_Recall": recall_score(y_test, y_pred),

            "Test_F1": f1_score(y_test, y_pred),

            "Test_ROC_AUC": roc_auc_score(y_test, y_prob),

            "Test_PR_AUC": average_precision_score(y_test, y_prob),

            "Training_Time": elapsed,

            "Number_of_Fits": (
                len(search.cv_results_["params"]) * N_SPLITS
            ),

            "Selected_Features": int(
                best_model.named_steps["selector"]
                .get_support().sum()
            )
        }

        optimization_results.append(metrics)

        best_estimators[
            f"{model_name}_{search_name}"
        ] = best_model

        print("Best CV ROC-AUC:", round(search.best_score_, 4))
        print("Test ROC-AUC:", round(metrics["Test_ROC_AUC"], 4))
        print("Test F1:", round(metrics["Test_F1"], 4))
        print("Execution Time:", round(elapsed, 2), "seconds")
        print("Best Parameters:", search.best_params_)


# ==========================================================
# 8. EXPERIMENT COMPARISON
# ==========================================================

results_df = pd.DataFrame(optimization_results)

results_df = results_df.sort_values(
    by="CV_ROC_AUC",
    ascending=False
).reset_index(drop=True)

print("\n[EXPERIMENT COMPARISON]")
print(results_df.round(4).to_string(index=False))


# ==========================================================
# 9. BEST MODEL
# ==========================================================

best_result = results_df.iloc[0]

best_key = (
    f"{best_result['Model']}_{best_result['Search']}"
)

best_model = best_estimators[best_key]

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("\nBEST MODEL:", best_key)


# ==========================================================
# 10. CLASSIFICATION REPORT
# ==========================================================

report = classification_report(
    y_test,
    y_pred,
    target_names=dataset.target_names,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

print("\n[CLASSIFICATION REPORT]")
print(report_df.round(4))


# ==========================================================
# 11. MULTI-METRIC CROSS VALIDATION
# ==========================================================

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

cv_results = cross_validate(
    best_model,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring,
    n_jobs=-1,
    return_train_score=True
)

cv_summary = pd.DataFrame({

    "Metric": list(scoring.keys()),

    "Train_Mean": [
        cv_results[f"train_{metric}"].mean()
        for metric in scoring
    ],

    "Validation_Mean": [
        cv_results[f"test_{metric}"].mean()
        for metric in scoring
    ],

    "Validation_Std": [
        cv_results[f"test_{metric}"].std()
        for metric in scoring
    ]
})

print("\n[CROSS VALIDATION SUMMARY]")
print(cv_summary.round(4).to_string(index=False))


# ==========================================================
# 12. FEATURE SELECTION
# ==========================================================

selector = best_model.named_steps["selector"]

feature_analysis = pd.DataFrame({

    "Feature": feature_names,
    "F_Score": selector.scores_,
    "P_Value": selector.pvalues_,
    "Selected": selector.get_support()

}).sort_values(
    by="F_Score",
    ascending=False
)

print("\n[FEATURE SELECTION]")
print(feature_analysis.to_string(index=False))


# ==========================================================
# 13. PERMUTATION IMPORTANCE
# ==========================================================

print("\n[PERMUTATION IMPORTANCE]")

permutation = permutation_importance(
    best_model,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=20,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

importance_df = pd.DataFrame({

    "Feature": feature_names,
    "Importance_Mean": permutation.importances_mean,
    "Importance_Std": permutation.importances_std

}).sort_values(
    by="Importance_Mean",
    ascending=False
)

print(importance_df.head(15).round(4).to_string(index=False))


# ==========================================================
# 14. CONFUSION MATRIX
# ==========================================================

fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True,
    fmt="d",
    cmap=sns.color_palette(
        ["#111827", "#00E5FF", "#3B82F6", "#A855F7"],
        as_cmap=True
    ),
    linewidths=2,
    linecolor=COLORS["background"],
    square=True,
    cbar=True,
    annot_kws={
        "size": 20,
        "weight": "bold",
        "color": "white"
    },
    xticklabels=dataset.target_names,
    yticklabels=dataset.target_names,
    ax=ax
)

style_axis(
    ax,
    "CONFUSION MATRIX",
    "Predicted Class",
    "Actual Class"
)

save_chart("01_confusion_matrix.png")


# ==========================================================
# 15. ROC CURVE
# ==========================================================

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

roc_auc = roc_auc_score(y_test, y_prob)

fig, ax = plt.subplots(figsize=(11, 8))

ax.plot(
    fpr,
    tpr,
    color=COLORS["cyan"],
    linewidth=3,
    label=f"Best Model | AUC = {roc_auc:.4f}"
)

ax.fill_between(
    fpr,
    tpr,
    color=COLORS["cyan"],
    alpha=0.12
)

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color=COLORS["muted"],
    linewidth=1.5,
    label="Random Classifier"
)

style_axis(
    ax,
    "ROC CURVE ANALYSIS",
    "False Positive Rate",
    "True Positive Rate"
)

ax.legend(loc="lower right")

save_chart("02_roc_curve.png")


# ==========================================================
# 16. PRECISION RECALL CURVE
# ==========================================================

precision, recall, thresholds = precision_recall_curve(
    y_test,
    y_prob
)

fig, ax = plt.subplots(figsize=(11, 8))

ax.plot(
    recall,
    precision,
    color=COLORS["pink"],
    linewidth=3
)

ax.fill_between(
    recall,
    precision,
    color=COLORS["pink"],
    alpha=0.12
)

style_axis(
    ax,
    "PRECISION - RECALL CURVE",
    "Recall",
    "Precision"
)

save_chart("03_precision_recall.png")


# ==========================================================
# 17. LEARNING CURVE
# ==========================================================

train_sizes, train_scores, validation_scores = learning_curve(
    best_model,
    X_train,
    y_train,
    cv=cv,
    scoring="roc_auc",
    train_sizes=np.linspace(0.2, 1.0, 5),
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)

validation_mean = validation_scores.mean(axis=1)
validation_std = validation_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(11, 8))

ax.plot(
    train_sizes,
    train_mean,
    marker="o",
    markersize=9,
    linewidth=3,
    color=COLORS["cyan"],
    label="Training Score"
)

ax.plot(
    train_sizes,
    validation_mean,
    marker="s",
    markersize=9,
    linewidth=3,
    color=COLORS["purple"],
    label="Validation Score"
)

ax.fill_between(
    train_sizes,
    train_mean - train_std,
    train_mean + train_std,
    color=COLORS["cyan"],
    alpha=0.12
)

ax.fill_between(
    train_sizes,
    validation_mean - validation_std,
    validation_mean + validation_std,
    color=COLORS["purple"],
    alpha=0.12
)

style_axis(
    ax,
    "LEARNING CURVE ANALYSIS",
    "Training Samples",
    "ROC-AUC"
)

ax.legend()

save_chart("04_learning_curve.png")


# ==========================================================
# 18. F-CLASSIF FEATURE SCORES
# ==========================================================

top_scores = feature_analysis.head(15).sort_values(
    by="F_Score"
)

fig, ax = plt.subplots(figsize=(12, 9))

bars = ax.barh(
    top_scores["Feature"],
    top_scores["F_Score"],
    color=COLORS["purple"],
    edgecolor=COLORS["cyan"],
    linewidth=0.7,
    alpha=0.9
)

for bar in bars:

    width = bar.get_width()

    ax.text(
        width,
        bar.get_y() + bar.get_height() / 2,
        f" {width:.1f}",
        va="center",
        color=COLORS["white"],
        fontsize=9
    )

style_axis(
    ax,
    "TOP 15 FEATURES - ANOVA F-TEST",
    "F-Score",
    "Features"
)

save_chart("05_feature_selection.png")


# ==========================================================
# 19. PERMUTATION IMPORTANCE
# ==========================================================

top_importance = importance_df.head(15).sort_values(
    by="Importance_Mean"
)

fig, ax = plt.subplots(figsize=(12, 9))

ax.barh(
    top_importance["Feature"],
    top_importance["Importance_Mean"],
    xerr=top_importance["Importance_Std"],
    color=COLORS["cyan"],
    edgecolor=COLORS["blue"],
    linewidth=0.8,
    capsize=4,
    alpha=0.85
)

style_axis(
    ax,
    "PERMUTATION FEATURE IMPORTANCE",
    "Mean ROC-AUC Score Decrease",
    "Features"
)

save_chart("06_permutation_importance.png")


# ==========================================================
# 20. MODEL ROC-AUC COMPARISON - CORRECTED
# ==========================================================

fig, ax = plt.subplots(figsize=(12, 8))

sns.barplot(
    data=results_df,
    x="Model",
    y="Test_ROC_AUC",
    hue="Search",
    palette=[
        COLORS["cyan"],
        COLORS["purple"]
    ],
    ax=ax
)

style_axis(
    ax,
    "MODEL ROC-AUC COMPARISON",
    "Machine Learning Models",
    "Test ROC-AUC"
)

ax.set_ylim(0.85, 1.01)

ax.legend(
    title="Optimization Method",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

save_chart("07_model_comparison.png")


# ==========================================================
# 21. GRID SEARCH VS RANDOMIZED SEARCH
# ==========================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16, 7)
)

sns.barplot(
    data=results_df,
    x="Model",
    y="CV_ROC_AUC",
    hue="Search",
    palette=[
        COLORS["cyan"],
        COLORS["purple"]
    ],
    ax=axes[0]
)

style_axis(
    axes[0],
    "CROSS VALIDATION ROC-AUC",
    "Model",
    "ROC-AUC"
)

sns.barplot(
    data=results_df,
    x="Model",
    y="Training_Time",
    hue="Search",
    palette=[
        COLORS["pink"],
        COLORS["orange"]
    ],
    ax=axes[1]
)

style_axis(
    axes[1],
    "OPTIMIZATION EXECUTION TIME",
    "Model",
    "Seconds"
)

for axis in axes:
    axis.legend(title="Search Algorithm")

save_chart("08_search_comparison.png")


# ==========================================================
# 22. CROSS VALIDATION METRICS
# ==========================================================

cv_plot = cv_summary.melt(
    id_vars="Metric",
    value_vars=[
        "Train_Mean",
        "Validation_Mean"
    ],
    var_name="Dataset",
    value_name="Score"
)

fig, ax = plt.subplots(figsize=(12, 7))

sns.barplot(
    data=cv_plot,
    x="Metric",
    y="Score",
    hue="Dataset",
    palette=[
        COLORS["cyan"],
        COLORS["purple"]
    ],
    ax=ax
)

style_axis(
    ax,
    "CROSS VALIDATION METRIC ANALYSIS",
    "Evaluation Metric",
    "Score"
)

ax.set_ylim(0, 1.05)
ax.legend(title="Data")

save_chart("09_cross_validation.png")


# ==========================================================
# 23. PREDICTION BENCHMARK
# ==========================================================

start = time.perf_counter()

for _ in range(100):
    best_model.predict(X_test)

prediction_time = time.perf_counter() - start

print("\n[PREDICTION BENCHMARK]")

print(
    "100 prediction calls:",
    round(prediction_time, 5),
    "seconds"
)


# ==========================================================
# 24. SAVE MODEL
# ==========================================================

joblib.dump(
    best_model,
    BASE_DIR / "best_ml_optimization_model.joblib"
)


# ==========================================================
# 25. EXPORT RESULTS
# ==========================================================

results_df.to_csv(
    BASE_DIR / "optimization_results.csv",
    index=False
)

feature_analysis.to_csv(
    BASE_DIR / "feature_selection_results.csv",
    index=False
)

importance_df.to_csv(
    BASE_DIR / "permutation_importance.csv",
    index=False
)

cv_summary.to_csv(
    BASE_DIR / "cross_validation_results.csv",
    index=False
)

report_df.to_csv(
    BASE_DIR / "classification_report.csv"
)


# ==========================================================
# 26. EXPERIMENT METADATA
# ==========================================================

metadata = {

    "project": "IMOFSE",

    "best_model": best_key,

    "best_cv_roc_auc": float(
        best_result["CV_ROC_AUC"]
    ),

    "test_roc_auc": float(roc_auc),

    "test_f1": float(
        f1_score(y_test, y_pred)
    ),

    "selected_features": int(
        selector.get_support().sum()
    ),

    "total_features": len(feature_names),

    "cv_folds": N_SPLITS,

    "random_state": RANDOM_STATE,

    "prediction_benchmark_100_calls": prediction_time,

    "experiment_duration": round(
        time.time() - experiment_start,
        2
    )
}

with open(
    BASE_DIR / "experiment_metadata.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=4,
        ensure_ascii=False
    )


# ==========================================================
# 27. FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 70)
print("IMOFSE - EXPERIMENT COMPLETED")
print("=" * 70)

print("\nBEST MODEL:", best_key)

print(
    "CV ROC-AUC:",
    round(best_result["CV_ROC_AUC"], 4)
)

print(
    "TEST ROC-AUC:",
    round(roc_auc, 4)
)

print(
    "TEST F1:",
    round(f1_score(y_test, y_pred), 4)
)

print("\nALL RESULTS SAVED IN:")
print(BASE_DIR)

print("\nGENERATED FILES:")

generated_files = [
    "best_ml_optimization_model.joblib",
    "optimization_results.csv",
    "feature_selection_results.csv",
    "permutation_importance.csv",
    "cross_validation_results.csv",
    "classification_report.csv",
    "experiment_metadata.json",
    "01_confusion_matrix.png",
    "02_roc_curve.png",
    "03_precision_recall.png",
    "04_learning_curve.png",
    "05_feature_selection.png",
    "06_permutation_importance.png",
    "07_model_comparison.png",
    "08_search_comparison.png",
    "09_cross_validation.png"
]

for filename in generated_files:

    file_path = BASE_DIR / filename

    if file_path.exists():
        print("✓", filename)

print("\nTOTAL EXECUTION TIME:",
      round(time.time() - experiment_start, 2),
      "seconds")

print("\nSUCCESS!")