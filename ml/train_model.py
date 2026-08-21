from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "landmarks.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "sign_classifier.joblib"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"لم يتم العثور على ملف البيانات:\n{DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)

    print("\nعدد العينات لكل فئة:")
    print(data["label"].value_counts())

    if data["label"].nunique() < 2:
        raise ValueError("نحتاج فئتين على الأقل لتدريب النموذج.")

    # المدخلات: نقاط اليد
    x = data.drop(columns=["label"])

    # المخرجات: اسم الإشارة
    y = data["label"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"\nAccuracy: {accuracy:.2%}\n")

    print("Classification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print(f"\nتم حفظ النموذج هنا:\n{MODEL_PATH}")


if __name__ == "__main__":
    main()