import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

# Nguong chat luong cua lab nay la f1_score, KHONG phai accuracy.
# Ly do: bo du lieu Adult co ty le lop 75/25. Mot mo hinh doan bua
# "thu nhap thap" cho moi mau da dat accuracy 0.75 ma khong hoc duoc gi.
F1_THRESHOLD = 0.65


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho GradientBoostingClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia (holdout).

    Tra ve:
        f1 (float): diem F1 cua lop duong (thu nhap > 50K) tren tap holdout.
    """

    # Bonus 1 (neu co cau hinh): Thiet lap remote tracking URI tu bien moi truong
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"MLflow connected to remote server: {tracking_uri}")

    # TODO 1: Doc du lieu huan luyen va danh gia
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # TODO 2: Tach dac trung (X) va nhan (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # Bonus 5: Kiem tra phan phoi va canh bao lech lac du lieu (Data Drift)
    pos_count = int((y_train == 1).sum())
    total_count = len(y_train)
    pos_rate = float(pos_count / total_count) if total_count > 0 else 0.0
    ref_rate = 0.248
    drift_delta = abs(pos_rate - ref_rate)

    print(f"Bonus 5 - Ty le lop duong trong tap train: {pos_rate:.4f} (Tham chieu: {ref_rate:.4f})")
    if drift_delta > 0.05:
        print(f"⚠️ CANH BAO: Phat hien lech lac phan phoi du lieu! Ty le hien tai ({pos_rate:.4f}) lech {drift_delta*100:.2f}% (> 5%) so voi moc tham chieu ({ref_rate:.4f}).")
    else:
        print(f"Bonus 5 - Phan phoi du lieu on dinh (do lech {drift_delta*100:.2f}% <= 5%).")

    with mlflow.start_run():

        # TODO 3: Ghi nhan cac sieu tham so
        mlflow.log_params(params)

        # TODO 4: Khoi tao va huan luyen GradientBoostingClassifier
        # Goi y: su dung random_state=42 de dam bao tinh tai tao
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # TODO 5: Du doan tren tap holdout va tinh chi so
        # Chu y: f1_score o day tinh cho LOP DUONG (target = 1), khong dung average.
        preds = model.predict(X_eval)
        f1 = float(f1_score(y_eval, preds, zero_division=0))
        acc = float(accuracy_score(y_eval, preds))

        # Bonus 2: Dieu chinh nguong quyet dinh toi uu (Decision Threshold Tuning)
        eval_probs = model.predict_proba(X_eval)[:, 1]
        thresholds = np.arange(0.1, 0.95, 0.05)
        best_threshold = 0.5
        best_f1 = f1

        for th in thresholds:
            th = round(float(th), 2)
            th_preds = (eval_probs >= th).astype(int)
            th_f1 = float(f1_score(y_eval, th_preds, zero_division=0))
            if th_f1 > best_f1:
                best_f1 = th_f1
                best_threshold = th

        print(f"Bonus 2 - Nguong toi uu: {best_threshold} (F1: {best_f1:.4f} | F1 mac dinh tai 0.5: {f1:.4f})")

        # Bonus 3: Bao cao Precision / Recall chi tiet va Confusion Matrix
        cm = confusion_matrix(y_eval, preds)
        prec_0 = float(precision_score(y_eval, preds, pos_label=0, zero_division=0))
        rec_0 = float(recall_score(y_eval, preds, pos_label=0, zero_division=0))
        prec_1 = float(precision_score(y_eval, preds, pos_label=1, zero_division=0))
        rec_1 = float(recall_score(y_eval, preds, pos_label=1, zero_division=0))
        clf_report = classification_report(y_eval, preds, target_names=["<=50K (0)", ">50K (1)"], zero_division=0)

        detail_text = (
            "==================================================\n"
            "BAO CAO CHI TIET MODEL INCOME (BONUS 3)\n"
            "==================================================\n\n"
            "1. CONFUSION MATRIX:\n"
            f"[[TN={cm[0,0]}, FP={cm[0,1]}],\n"
            f" [FN={cm[1,0]}, TP={cm[1,1]}]]\n\n"
            "2. METRICS THEO TUNG LOP:\n"
            f"- Lop 0 (<=50K):\n"
            f"    Precision: {prec_0:.4f}\n"
            f"    Recall:    {rec_0:.4f}\n"
            f"- Lop 1 (>50K):\n"
            f"    Precision: {prec_1:.4f}\n"
            f"    Recall:    {rec_1:.4f}\n\n"
            "3. CLASSIFICATION REPORT:\n"
            f"{clf_report}\n"
        )
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/detail.txt", "w") as f:
            f.write(detail_text)

        # TODO 6: Ghi nhan chi so vao MLflow
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("best_f1", best_f1)
        mlflow.log_metric("positive_class_rate", pos_rate)
        mlflow.sklearn.log_model(model, "model")

        # TODO 7: In ket qua ra man hinh
        print(f"F1: {f1:.4f} | Accuracy: {acc:.4f}")

        # TODO 8: Luu metrics ra file outputs/report.json
        # File nay duoc doc boi GitHub Actions o Buoc 2
        with open("outputs/report.json", "w") as f:
            json.dump({
                "f1_score": f1,
                "accuracy": acc,
                "best_threshold": best_threshold,
                "best_f1": best_f1,
                "positive_class_rate": pos_rate,
            }, f, indent=2)

        # TODO 9: Luu mo hinh ra file models/model.joblib
        # File nay duoc upload len cloud storage o Buoc 2
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    # TODO 10: Tra ve f1
    return f1


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
