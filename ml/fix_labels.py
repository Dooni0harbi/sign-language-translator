from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "landmarks.csv"
BACKUP_PATH = BASE_DIR / "data" / "landmarks_backup.csv"


def main() -> None:
    data = pd.read_csv(CSV_PATH)

    # الاحتفاظ بنسخة احتياطية قبل التعديل
    data.to_csv(BACKUP_PATH, index=False)

    jeem_rows = data.index[data["label"] == "jeem"].tolist()

    if len(jeem_rows) != 200:
        raise ValueError(
            f"توقعت 200 عينة باسم jeem، لكن وجدت {len(jeem_rows)}."
        )

    # أول 50 كانت ج فعلًا
    data.loc[jeem_rows[0:50], "label"] = "jeem"

    # الـ50 التالية كانت د
    data.loc[jeem_rows[50:100], "label"] = "dal"

    # الـ50 التالية كانت ا
    data.loc[jeem_rows[100:150], "label"] = "alif"

    # آخر 50 كانت ن
    data.loc[jeem_rows[150:200], "label"] = "noon"

    data.to_csv(CSV_PATH, index=False)

    print("\nبعد التصحيح:")
    print(data["label"].value_counts())

    print(f"\nتم حفظ نسخة احتياطية هنا:\n{BACKUP_PATH}")


if __name__ == "__main__":
    main()