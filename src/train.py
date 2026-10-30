"""Train and test the obesity level model (script version of the notebook).

Steps: lock the test set, compare 4 models with 5-fold cross-validation,
tune the best one, test it once, and save the results and charts.

Run from the project folder:
    python src/train.py
    python src/train.py --data data/my_file.csv --out reports/script_run
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # draw charts without opening a window
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42

# the 7 weight levels, from lightest to heaviest
CLASS_ORDER = ["Insufficient_Weight", "Normal_Weight", "Overweight_Level_I",
               "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II",
               "Obesity_Type_III"]


def load_data(path):
    """Read the CSV file and make sure the target column is called NObesity."""
    data = pd.read_csv(path)
    if "NObeyesdad" in data.columns:
        data = data.rename(columns={"NObeyesdad": "NObesity"})
    return data


def build_preprocessor(X_train):
    """Fill missing values, scale number columns and one-hot encode text columns."""
    number_cols = X_train.select_dtypes(include="number").columns.tolist()
    text_cols = X_train.select_dtypes(exclude="number").columns.tolist()

    number_steps = Pipeline(steps=[
        ("fill", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    text_steps = Pipeline(steps=[
        ("fill", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("numbers", number_steps, number_cols),
        ("text", text_steps, text_cols),
    ])


def get_param_grid(best_name):
    """Small settings grid for the model that won the comparison."""
    if best_name == "Random Forest":
        return {"model__n_estimators": [100, 200],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 5]}
    if best_name == "Logistic Regression":
        return {"model__C": [0.1, 1, 10, 100]}
    if best_name == "Decision Tree":
        return {"model__max_depth": [5, 10, 15, None],
                "model__min_samples_leaf": [1, 2, 5]}
    return {"model__n_neighbors": [3, 5, 7, 9, 11],
            "model__weights": ["uniform", "distance"]}


def main():
    parser = argparse.ArgumentParser(description="Train the obesity level model.")
    parser.add_argument("--data", default="data/obesity_data.csv", help="path to the CSV file")
    parser.add_argument("--out", default="reports/script_run", help="folder for results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    # 1. load the data and remove Height and Weight (they give away the answer through BMI)
    data = load_data(args.data)
    data = data.drop(columns=["Height", "Weight"])
    X = data.drop(columns=["NObesity"])
    y = data["NObesity"]

    # 2. lock away the test set FIRST (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    print("Training rows:", len(X_train), "| Test rows:", len(X_test))

    preprocessor = build_preprocessor(X_train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # 3. baseline and 4 models, all checked with 5-fold cross-validation
    models = {
        "Baseline (most frequent)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(),
    }
    names, means, stds = [], [], []
    for name in models:
        pipe = Pipeline(steps=[("prep", preprocessor), ("model", models[name])])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro")
        names.append(name)
        means.append(scores.mean())
        stds.append(scores.std())
        print(f"{name:26s} mean macro F1 = {scores.mean():.3f} (std {scores.std():.3f})")

    results = pd.DataFrame({"Model": names, "Mean macro F1": means, "Std": stds})
    results = results.sort_values("Mean macro F1", ascending=False).round(3)
    results.to_csv(out_dir / "cv_results.csv", index=False)

    # 4. tune the best real model (not the baseline)
    best_name = results[results["Model"] != "Baseline (most frequent)"].iloc[0]["Model"]
    best_pipe = Pipeline(steps=[("prep", preprocessor), ("model", models[best_name])])
    grid = GridSearchCV(best_pipe, get_param_grid(best_name), cv=cv,
                        scoring="f1_macro", n_jobs=-1)
    grid.fit(X_train, y_train)
    print("Best model:", best_name, "| best settings:", grid.best_params_)

    # 5. test ONE time on the locked test set
    final_model = grid.best_estimator_
    y_pred = final_model.predict(X_test)
    test_f1 = f1_score(y_test, y_pred, average="macro")
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test macro F1 = {test_f1:.3f} | Test accuracy = {test_accuracy:.3f}")

    # 6. save results
    metrics = {
        "best_model": best_name,
        "best_params": grid.best_params_,
        "cv_macro_f1_after_tuning": round(grid.best_score_, 3),
        "test_macro_f1": round(test_f1, 3),
        "test_accuracy": round(test_accuracy, 3),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred, digits=3))
    joblib.dump(final_model, Path("models") / "obesity_model.joblib")

    # 7. save the model comparison chart and the confusion matrix
    plt.figure(figsize=(9, 5))
    sns.barplot(data=results, x="Mean macro F1", y="Model", color="steelblue")
    plt.title("Model Comparison (5-Fold Cross-Validation)")
    plt.xlabel("Mean macro F1-score (higher is better)")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(out_dir / "model_comparison.png", dpi=150)
    plt.close()

    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER)
    plt.title("Confusion Matrix on the Test Set")
    plt.xlabel("Predicted level")
    plt.ylabel("True level")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    print("Results saved in", out_dir)


if __name__ == "__main__":
    main()
