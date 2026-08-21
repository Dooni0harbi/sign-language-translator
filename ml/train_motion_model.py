from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent

MOTION_DATA_DIR = BASE_DIR / "motion_data"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "motion_classifier.joblib"

SEQUENCE_LENGTH = 30
FEATURES_PER_FRAME = 63

LABELS = [
    "esme_motion",
    "other_motion",
]


def load_motion_data():
    features = []
    labels = []

    for label in LABELS:
        label_dir = MOTION_DATA_DIR / label

        if not label_dir.exists():
            raise FileNotFoundError(
                f"المجلد غير موجود:\n{label_dir}"
            )

        sequence_files = sorted(
            label_dir.glob("sequence_*.npy")
        )

        if not sequence_files:
            raise ValueError(
                f"لا توجد عينات داخل:\n{label_dir}"
            )

        print(
            f"{label}: {len(sequence_files)} sequences"
        )

        for file_path in sequence_files:
            sequence = np.load(file_path)

            expected_shape = (
                SEQUENCE_LENGTH,
                FEATURES_PER_FRAME,
            )

            if sequence.shape != expected_shape:
                print(
                    f"تم تجاهل {file_path.name} "
                    f"لأن أبعاده {sequence.shape}"
                )
                continue

            # نحول 30 × 63 إلى صف واحد
            flattened_sequence = sequence.flatten()

            features.append(flattened_sequence)
            labels.append(label)

    if not features:
        raise ValueError("لم يتم تحميل أي بيانات صالحة.")

    return (
        np.array(features, dtype=np.float32),
        np.array(labels),
    )


def main():
    x, y = load_motion_data()

    print("\nإجمالي العينات:", len(y))

    unique_labels, counts = np.unique(
        y,
        return_counts=True,
    )

    print("\nعدد العينات لكل فئة:")

    for label, count in zip(unique_labels, counts):
        print(f"{label}: {count}")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    print("\nجاري تدريب نموذج الحركة...")

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print(f"\nAccuracy: {accuracy:.2%}\n")

    print("Classification report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print(
        f"\nتم حفظ نموذج الحركة هنا:\n"
        f"{MODEL_PATH}"
    )


if __name__ == "__main__":
    main()