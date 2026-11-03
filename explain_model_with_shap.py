"""
Explain a Machine Learning Model with SHAP
------------------------------------------
Trains a Random Forest on the breast cancer dataset (built into scikit-learn),
then uses SHAP to explain the model globally (which features matter most)
and locally (why the model made one specific prediction).

Requirements:  pip install scikit-learn shap matplotlib pandas
Run:           python explain_model_with_shap.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save plots to files without opening windows
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

OUT_DIR = Path("plots")
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------
# Step 1: Load the data
# ---------------------------------------------------------------
def load_data():
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target  # target: 0 = malignant, 1 = benign
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    counts = y.value_counts().sort_index()
    print(f"Classes: {', '.join(f'{n}={c}' for n, c in zip(data.target_names, counts))}")
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ---------------------------------------------------------------
# Step 2: Train the model
# ---------------------------------------------------------------
def train_model(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, preds):.3f}")
    print(classification_report(y_test, preds, target_names=["malignant", "benign"]))
    return model


# ---------------------------------------------------------------
# Step 3: Compute SHAP values
# ---------------------------------------------------------------
def compute_shap(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    # For a binary classifier, SHAP returns one set of values per class.
    # We keep class 1 ("benign"): positive values push towards benign.
    return shap_values[:, :, 1]


# ---------------------------------------------------------------
# Step 4: Global explanations
# ---------------------------------------------------------------
def global_plots(sv):
    shap.plots.bar(sv, max_display=10, show=False)
    plt.title("Top 10 features by mean |SHAP value|")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_bar.png", dpi=150)
    plt.close()

    shap.plots.beeswarm(sv, max_display=10, show=False)
    plt.title("How each feature pushes predictions")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_beeswarm.png", dpi=150)
    plt.close()

    importance = (pd.Series(abs(sv.values).mean(axis=0), index=sv.feature_names)
                    .sort_values(ascending=False))
    print("Top 5 features by mean |SHAP|:")
    print(importance.head(5).round(4).to_string())


# ---------------------------------------------------------------
# Step 5: Local explanation for one patient
# ---------------------------------------------------------------
def explain_one(model, sv, X_test, y_test, idx=0):
    proba = model.predict_proba(X_test.iloc[[idx]])[0, 1]
    actual = "benign" if y_test.iloc[idx] == 1 else "malignant"
    print(f"\nSample {idx}: predicted P(benign) = {proba:.3f}, actual = {actual}")
    print(f"Base value (average prediction): {sv.base_values[idx]:.3f}")

    contrib = pd.Series(sv.values[idx], index=sv.feature_names)
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(5)
    print("Top 5 contributions for this sample:")
    print(top.round(4).to_string())

    shap.plots.waterfall(sv[idx], max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"shap_waterfall_sample{idx}.png", dpi=150)
    plt.close()


def main():
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train, X_test, y_test)
    sv = compute_shap(model, X_test)
    global_plots(sv)
    # Pick the first malignant case in the test set: the interesting one to explain
    idx = int((y_test.values == 0).argmax())
    explain_one(model, sv, X_test, y_test, idx=idx)
    print(f"\nPlots saved in: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
