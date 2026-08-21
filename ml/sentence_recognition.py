import time
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np


# ---------------------------------------------------------
# المسارات والإعدادات
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

HAND_MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"
CLASSIFIER_PATH = BASE_DIR / "models" / "sign_classifier.joblib"

CAMERA_INDEX = 0

# أقل نسبة ثقة لقبول التنبؤ
CONFIDENCE_THRESHOLD = 0.75

# مدة تثبيت الإشارة قبل إضافتها للجملة
HOLD_DURATION = 0.8


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


DISPLAY_NAMES = {
    "ana": "ANA",
    "esme": "ESME",
    "waw": "WAW",
    "jeem": "JEEM",
    "dal": "DAL",
    "alif": "ALIF",
    "noon": "NOON",
}


# الترتيب الصحيح للجملة المستهدفة
TARGET_SEQUENCE = [
    "ana",
    "esme",
    "waw",
    "jeem",
    "dal",
    "alif",
    "noon",
]


# ---------------------------------------------------------
# تجهيز نقاط اليد
# ---------------------------------------------------------

def normalize_landmarks(landmarks) -> list[float]:
    wrist = landmarks[0]
    features = []

    for landmark in landmarks:
        features.extend([
            landmark.x - wrist.x,
            landmark.y - wrist.y,
            landmark.z - wrist.z,
        ])

    return features


def draw_hand(frame, landmarks) -> None:
    height, width, _ = frame.shape
    points = []

    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((x, y))

    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            (255, 255, 255),
            2,
        )

    for point in points:
        cv2.circle(
            frame,
            point,
            5,
            (0, 255, 0),
            -1,
        )


def get_sequence_text(sequence: list[str]) -> str:
    return " > ".join(
        DISPLAY_NAMES.get(label, label.upper())
        for label in sequence
    )


def print_translation() -> None:
    print("\n" + "=" * 45)
    print("Arabic:  أنا اسمي وجدان")
    print("English: My name is Wejdan")
    print("German:  Ich heiße Wejdan")
    print("=" * 45 + "\n")


# ---------------------------------------------------------
# البرنامج الرئيسي
# ---------------------------------------------------------

def main() -> None:
    if not HAND_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Hand model was not found:\n{HAND_MODEL_PATH}"
        )

    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            f"Classifier was not found:\n{CLASSIFIER_PATH}"
        )

    classifier = joblib.load(CLASSIFIER_PATH)

    base_options = mp.tasks.BaseOptions(
        model_asset_path=str(HAND_MODEL_PATH)
    )

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        raise RuntimeError("تعذر فتح الكاميرا.")

    start_time = time.perf_counter()

    sentence_sequence = []

    current_candidate = None
    candidate_start_time = None

    # يمنع إضافة نفس الإشارة باستمرار
    waiting_for_change = False
    last_added_label = None

    sentence_completed = False

    with mp.tasks.vision.HandLandmarker.create_from_options(
        options
    ) as detector:

        while True:
            success, frame = camera.read()

            if not success:
                print("تعذر قراءة الكاميرا.")
                break

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            timestamp_ms = int(
                (time.perf_counter() - start_time) * 1000
            )

            result = detector.detect_for_video(
                mp_image,
                timestamp_ms,
            )

            prediction_text = "No hand"
            progress = 0.0

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]

                draw_hand(frame, landmarks)

                features = normalize_landmarks(landmarks)

                features_array = np.array(
                    features,
                    dtype=np.float32,
                ).reshape(1, -1)

                probabilities = classifier.predict_proba(
                    features_array
                )[0]

                best_index = int(np.argmax(probabilities))
                confidence = float(probabilities[best_index])
                predicted_label = classifier.classes_[best_index]

                if confidence >= CONFIDENCE_THRESHOLD:
                    display_label = DISPLAY_NAMES.get(
                        predicted_label,
                        predicted_label.upper(),
                    )

                    prediction_text = (
                        f"{display_label} | "
                        f"{confidence * 100:.1f}%"
                    )

                    # إذا تغيرت الإشارة نسمح بالتسجيل مجددًا
                    if (
                        waiting_for_change
                        and predicted_label != last_added_label
                    ):
                        waiting_for_change = False

                    if not waiting_for_change:
                        if predicted_label != current_candidate:
                            current_candidate = predicted_label
                            candidate_start_time = time.perf_counter()

                        elif candidate_start_time is not None:
                            elapsed = (
                                time.perf_counter()
                                - candidate_start_time
                            )

                            progress = min(
                                elapsed / HOLD_DURATION,
                                1.0,
                            )

                            if elapsed >= HOLD_DURATION:
                                sentence_sequence.append(
                                    predicted_label
                                )

                                print(
                                    "Added:",
                                    DISPLAY_NAMES.get(
                                        predicted_label,
                                        predicted_label,
                                    ),
                                )

                                print(
                                    "Sequence:",
                                    get_sequence_text(
                                        sentence_sequence
                                    ),
                                )

                                last_added_label = predicted_label
                                waiting_for_change = True
                                current_candidate = None
                                candidate_start_time = None

                                if (
                                    sentence_sequence
                                    == TARGET_SEQUENCE
                                ):
                                    sentence_completed = True
                                    print_translation()

                else:
                    prediction_text = (
                        f"Unknown | {confidence * 100:.1f}%"
                    )

                    current_candidate = None
                    candidate_start_time = None

            else:
                # إزالة اليد تسمح بتسجيل الإشارة التالية
                current_candidate = None
                candidate_start_time = None
                waiting_for_change = False

            # -------------------------------------------------
            # النصوص الظاهرة على الكاميرا
            # -------------------------------------------------

            cv2.putText(
                frame,
                prediction_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            sequence_text = get_sequence_text(
                sentence_sequence
            )

            cv2.putText(
                frame,
                f"Sequence: {sequence_text}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            # شريط تثبيت الإشارة
            bar_x = 20
            bar_y = 105
            bar_width = 250
            bar_height = 18

            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + bar_width, bar_y + bar_height),
                (255, 255, 255),
                2,
            )

            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (
                    bar_x + int(bar_width * progress),
                    bar_y + bar_height,
                ),
                (0, 255, 0),
                -1,
            )

            if sentence_completed:
                cv2.putText(
                    frame,
                    "My name is Wejdan",
                    (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    "Ich heisse Wejdan",
                    (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

            cv2.putText(
                frame,
                "C: Clear | B: Backspace | Q: Quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "Arabic Sign Sentence Recognition",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("c"):
                sentence_sequence.clear()
                sentence_completed = False
                waiting_for_change = False
                last_added_label = None
                current_candidate = None
                candidate_start_time = None

                print("Sequence cleared.")

            elif key == ord("b"):
                if sentence_sequence:
                    removed = sentence_sequence.pop()
                    sentence_completed = False
                    waiting_for_change = False

                    print(
                        "Removed:",
                        DISPLAY_NAMES.get(
                            removed,
                            removed,
                        ),
                    )

            elif key == ord("q"):
                break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()