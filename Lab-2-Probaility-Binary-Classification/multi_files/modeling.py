import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    )

    model.fit(X_train, y_train)
    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    return y_pred


def compare_thresholds(model, X_test, y_test):
    probabilities = model.predict_proba(X_test)

    results = pd.DataFrame({
        "Actual_class": y_test.values,
        "P_malignant": probabilities[:, 0],
        "P_benign": probabilities[:, 1],
    })

    results["Actual_label"] = results["Actual_class"].map({0: "malignant", 1: "benign"})

    print("\nFirst 10 Probability Results:\n")
    print(results[["Actual_class", "Actual_label", "P_malignant", "P_benign"]].head(10))

    for threshold in [0.3, 0.5, 0.7]:
        predicted_malignant = (results["P_malignant"] >= threshold).astype(int)
        results[f"Predicted_malignant_{threshold}"] = predicted_malignant
        results[f"Predicted_label_{threshold}"] = predicted_malignant.map({1: "malignant", 0: "benign"})

        y_pred_threshold = (probabilities[:, 0] >= threshold).astype(int)

        print(f"\nConfusion Matrix with Threshold {threshold}:\n")
        print(confusion_matrix(y_test, y_pred_threshold))

        print(f"\nSample Predictions for Threshold {threshold}:\n")
        print(
            results[
                [
                    "Actual_label",
                    "P_malignant",
                    "P_benign",
                    f"Predicted_label_{threshold}",
                ]
            ].head(10)
        )

    return results


def print_metrics_for_thresholds(model, X_test, y_test):
    probabilities = model.predict_proba(X_test)
    y_true_malignant = (y_test == 0).astype(int)

    for threshold in [0.3, 0.5, 0.7]:
        y_pred_malignant = (probabilities[:, 0] >= threshold).astype(int)

        precision = precision_score(y_true_malignant, y_pred_malignant, zero_division=0)
        recall = recall_score(y_true_malignant, y_pred_malignant, zero_division=0)
        f1 = f1_score(y_true_malignant, y_pred_malignant, zero_division=0)
        accuracy = accuracy_score(y_true_malignant, y_pred_malignant)

        print(f"\nMetrics for Threshold {threshold}:")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"\nClassification Report for Threshold {threshold}:\n")
        print(classification_report(y_true_malignant, y_pred_malignant, target_names=["benign", "malignant"]))


def generate_report_table(model, X_test, y_test):
    probabilities = model.predict_proba(X_test)
    y_true_malignant = (y_test == 0).astype(int)

    report_data = []

    for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
        y_pred_malignant = (probabilities[:, 0] >= threshold).astype(int)

        precision = precision_score(y_true_malignant, y_pred_malignant, zero_division=0)
        recall = recall_score(y_true_malignant, y_pred_malignant, zero_division=0)
        f1 = f1_score(y_true_malignant, y_pred_malignant, zero_division=0)
        accuracy = accuracy_score(y_true_malignant, y_pred_malignant)
        tn, fp, fn, tp = confusion_matrix(y_true_malignant, y_pred_malignant).ravel()

        report_data.append({
            "Threshold": threshold,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Accuracy": accuracy,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
        })

    report_df = pd.DataFrame(report_data)
    print("\nClassification Report and Confusion Matrix for Different Thresholds:\n")
    print(report_df.to_string(index=False))
    return report_df
