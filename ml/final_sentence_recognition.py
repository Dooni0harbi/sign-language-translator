import json
import time
from collections import deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np


# =========================================================
# المسارات والإعدادات
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

HAND_MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"
STATIC_MODEL_PATH = BASE_DIR / "models" / "sign_classifier.joblib"
MOTION_MODEL_PATH = BASE_DIR / "models" / "motion_classifier.joblib"

# ملف النتيجة الذي ستقرأه واجهة الويب
RESULT_PATH = BASE_DIR / "translation_result.json"

CAMERA_INDEX = 0

# قبول الإشارات الثابتة من 85% فأعلى
STATIC_CONFIDENCE_THRESHOLD = 0.85

# نسبة قبول حركة ESME
MOTION_CONFIDENCE_THRESHOLD = 0.60

# عدد الإطارات المستخدمة للحركة
SEQUENCE_LENGTH = 30

# عدد الإطارات الصحيحة المطلوبة لقبول الحرف
REQUIRED_STABLE_FRAMES = 15

# منع إضافة حركة ESME بسرعة أكثر من مرة
MOTION_COOLDOWN = 2.0

# شروط الحركة الأفقية
MIN_HORIZONTAL_DISPLACEMENT = 0.03
MIN_HORIZONTAL_RANGE = 0.05


# =========================================================
# توصيلات نقاط اليد
# =========================================================

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]


# =========================================================
# أسماء الإشارات وترتيب الجملة
# =========================================================

DISPLAY_NAMES = {
    "ana": "ANA",
    "waw": "WAW",
    "jeem": "JEEM",
    "dal": "DAL",
    "alif": "ALIF",
    "noon": "NOON",
    "esme_motion": "ESME",
}


TARGET_SEQUENCE = [
    "ana",
    "esme_motion",
    "waw",
    "jeem",
    "dal",
    "alif",
    "noon",
]


# =========================================================
# الدوال المساعدة
# =========================================================

def normalize_label(label) -> str:
    """
    توحيد أسماء التصنيفات.
    """

    return str(label).strip().lower()


def normalize_landmarks(landmarks) -> list[float]:
    """
    تحويل نقاط اليد إلى قيم نسبية مقارنة بالمعصم.
    """

    wrist = landmarks[0]
    features = []

    for landmark in landmarks:
        features.extend(
            [
                landmark.x - wrist.x,
                landmark.y - wrist.y,
                landmark.z - wrist.z,
            ]
        )

    return features


def draw_hand(frame, landmarks) -> None:
    """
    رسم نقاط وخطوط اليد فوق صورة الكاميرا.
    """

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
    """
    تحويل قائمة الإشارات إلى نص ظاهر على الشاشة.
    """

    return " > ".join(
        DISPLAY_NAMES.get(label, label.upper())
        for label in sequence
    )


def get_expected_label(sentence_sequence: list[str]):
    """
    تحديد الإشارة التالية المطلوبة.
    """

    if len(sentence_sequence) >= len(TARGET_SEQUENCE):
        return None

    return TARGET_SEQUENCE[len(sentence_sequence)]


def save_translation_result() -> None:
    """
    حفظ الترجمة النهائية لتقرأها واجهة الويب.
    """

    result = {
        "completed": True,
        "arabic": "أنا اسمي وجدان",
        "english": "My name is Wejdan",
        "german": "Ich heiße Wejdan",
    }

    try:
        with RESULT_PATH.open(
            mode="w",
            encoding="utf-8",
        ) as result_file:
            json.dump(
                result,
                result_file,
                ensure_ascii=False,
                indent=2,
            )

        print("Translation saved:")
        print(result)

    except OSError as error:
        print(
            f"Could not save translation result: {error}"
        )


def clear_translation_result() -> None:
    """
    حذف النتيجة السابقة عند إعادة الجملة.
    """

    try:
        if RESULT_PATH.exists():
            RESULT_PATH.unlink()

    except OSError as error:
        print(
            f"Could not delete old result: {error}"
        )


# =========================================================
# البرنامج الرئيسي
# =========================================================

def main() -> None:
    required_files = [
        HAND_MODEL_PATH,
        STATIC_MODEL_PATH,
        MOTION_MODEL_PATH,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"الملف غير موجود:\n{file_path}"
            )

    # حذف أي ترجمة قديمة عند بداية التشغيل
    clear_translation_result()

    static_model = joblib.load(
        STATIC_MODEL_PATH
    )

    motion_model = joblib.load(
        MOTION_MODEL_PATH
    )

    base_options = mp.tasks.BaseOptions(
        model_asset_path=str(HAND_MODEL_PATH)
    )

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=(
            mp.tasks.vision.RunningMode.VIDEO
        ),
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    camera = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not camera.isOpened():
        raise RuntimeError(
            "تعذر فتح الكاميرا."
        )

    sequence_buffer = deque(
        maxlen=SEQUENCE_LENGTH
    )

    wrist_x_buffer = deque(
        maxlen=SEQUENCE_LENGTH
    )

    sentence_sequence = []

    stable_frame_count = 0

    last_motion_added_time = 0.0

    sentence_completed = False

    # يمنع تكرار حفظ نفس النتيجة كل فريم
    translation_saved = False

    start_time = time.perf_counter()

    with mp.tasks.vision.HandLandmarker.create_from_options(
        options
    ) as detector:

        while True:
            success, frame = camera.read()

            if not success:
                print(
                    "تعذر قراءة الصورة من الكاميرا."
                )
                break

            frame = cv2.flip(
                frame,
                1,
            )

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            timestamp_ms = int(
                (
                    time.perf_counter()
                    - start_time
                )
                * 1000
            )

            result = detector.detect_for_video(
                mp_image,
                timestamp_ms,
            )

            prediction_text = "Show one hand"
            confidence_text = ""
            movement_text = ""
            progress = 0.0

            expected_label = get_expected_label(
                sentence_sequence
            )

            # =================================================
            # اكتشاف اليد
            # =================================================

            if result.hand_landmarks:
                landmarks = (
                    result.hand_landmarks[0]
                )

                draw_hand(
                    frame,
                    landmarks,
                )

                features = normalize_landmarks(
                    landmarks
                )

                sequence_buffer.append(
                    features
                )

                wrist_x_buffer.append(
                    float(landmarks[0].x)
                )

                # =============================================
                # التعرف على حركة ESME
                # =============================================

                if expected_label == "esme_motion":
                    prediction_text = (
                        "Perform ESME motion"
                    )

                    if (
                        len(sequence_buffer)
                        == SEQUENCE_LENGTH
                        and len(wrist_x_buffer)
                        == SEQUENCE_LENGTH
                    ):
                        motion_array = np.array(
                            sequence_buffer,
                            dtype=np.float32,
                        ).reshape(1, -1)

                        motion_probabilities = (
                            motion_model.predict_proba(
                                motion_array
                            )[0]
                        )

                        motion_best_index = int(
                            np.argmax(
                                motion_probabilities
                            )
                        )

                        motion_confidence = float(
                            motion_probabilities[
                                motion_best_index
                            ]
                        )

                        motion_label = normalize_label(
                            motion_model.classes_[
                                motion_best_index
                            ]
                        )

                        wrist_values = np.array(
                            wrist_x_buffer,
                            dtype=np.float32,
                        )

                        horizontal_displacement = float(
                            wrist_values[-1]
                            - wrist_values[0]
                        )

                        horizontal_range = float(
                            wrist_values.max()
                            - wrist_values.min()
                        )

                        has_horizontal_displacement = (
                            abs(
                                horizontal_displacement
                            )
                            >= MIN_HORIZONTAL_DISPLACEMENT
                        )

                        has_horizontal_range = (
                            horizontal_range
                            >= MIN_HORIZONTAL_RANGE
                        )

                        confidence_text = (
                            "Motion confidence: "
                            f"{motion_confidence * 100:.1f}%"
                        )

                        movement_text = (
                            "Move X: "
                            f"{horizontal_displacement:.3f}"
                            " | Range: "
                            f"{horizontal_range:.3f}"
                        )

                        current_time = (
                            time.perf_counter()
                        )

                        if (
                            motion_label
                            == "esme_motion"
                            and motion_confidence
                            >= MOTION_CONFIDENCE_THRESHOLD
                            and has_horizontal_displacement
                            and has_horizontal_range
                            and current_time
                            - last_motion_added_time
                            >= MOTION_COOLDOWN
                        ):
                            sentence_sequence.append(
                                "esme_motion"
                            )

                            prediction_text = (
                                "ESME ADDED | "
                                f"{motion_confidence * 100:.1f}%"
                            )

                            print("Added: ESME")

                            print(
                                "Sequence:",
                                get_sequence_text(
                                    sentence_sequence
                                ),
                            )

                            last_motion_added_time = (
                                current_time
                            )

                            sequence_buffer.clear()
                            wrist_x_buffer.clear()

                            stable_frame_count = 0

                        else:
                            prediction_text = (
                                "ESME candidate | "
                                f"{motion_confidence * 100:.1f}%"
                            )

                # =============================================
                # التعرف على الإشارات الثابتة
                # =============================================

                elif expected_label is not None:
                    static_array = np.array(
                        features,
                        dtype=np.float32,
                    ).reshape(1, -1)

                    static_probabilities = (
                        static_model.predict_proba(
                            static_array
                        )[0]
                    )

                    static_best_index = int(
                        np.argmax(
                            static_probabilities
                        )
                    )

                    static_confidence = float(
                        static_probabilities[
                            static_best_index
                        ]
                    )

                    static_label = normalize_label(
                        static_model.classes_[
                            static_best_index
                        ]
                    )

                    normalized_expected_label = (
                        normalize_label(
                            expected_label
                        )
                    )

                    display_label = DISPLAY_NAMES.get(
                        static_label,
                        static_label.upper(),
                    )

                    expected_display = (
                        DISPLAY_NAMES.get(
                            normalized_expected_label,
                            normalized_expected_label.upper(),
                        )
                    )

                    prediction_text = (
                        f"{display_label} | "
                        f"{static_confidence * 100:.1f}%"
                    )

                    confidence_text = (
                        f"Required: {expected_display}"
                        " | Progress: "
                        f"{stable_frame_count}/"
                        f"{REQUIRED_STABLE_FRAMES}"
                    )

                    # قبول الحرف عندما:
                    # 1. يكون هو الحرف المطلوب
                    # 2. تكون الثقة 85% أو أعلى

                    if (
                        static_label
                        == normalized_expected_label
                        and static_confidence
                        >= STATIC_CONFIDENCE_THRESHOLD
                    ):
                        stable_frame_count += 1

                    else:
                        # لا ننقص العداد عند التذبذب
                        pass

                    progress = min(
                        stable_frame_count
                        / REQUIRED_STABLE_FRAMES,
                        1.0,
                    )

                    if (
                        stable_frame_count
                        >= REQUIRED_STABLE_FRAMES
                    ):
                        sentence_sequence.append(
                            normalized_expected_label
                        )

                        added_label = (
                            DISPLAY_NAMES.get(
                                normalized_expected_label,
                                normalized_expected_label.upper(),
                            )
                        )

                        print(
                            "Added:",
                            added_label,
                        )

                        print(
                            "Sequence:",
                            get_sequence_text(
                                sentence_sequence
                            ),
                        )

                        stable_frame_count = 0
                        progress = 0.0

                        sequence_buffer.clear()
                        wrist_x_buffer.clear()

                else:
                    prediction_text = (
                        "Sentence complete"
                    )

                    stable_frame_count = 0
                    progress = 0.0

            # =================================================
            # إذا اختفت اليد
            # =================================================

            else:
                sequence_buffer.clear()
                wrist_x_buffer.clear()

                progress = min(
                    stable_frame_count
                    / REQUIRED_STABLE_FRAMES,
                    1.0,
                )

            # =================================================
            # التحقق من اكتمال الجملة
            # =================================================

            sentence_completed = (
                sentence_sequence
                == TARGET_SEQUENCE
            )

            # حفظ النتيجة مرة واحدة فقط
            if (
                sentence_completed
                and not translation_saved
            ):
                save_translation_result()
                translation_saved = True

            expected_label = get_expected_label(
                sentence_sequence
            )

            if expected_label is None:
                expected_text = "DONE"

            else:
                expected_text = (
                    DISPLAY_NAMES.get(
                        expected_label,
                        expected_label.upper(),
                    )
                )

            # =================================================
            # عرض المعلومات
            # =================================================

            cv2.putText(
                frame,
                prediction_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Expected: {expected_text}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                confidence_text,
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                movement_text,
                (20, 132),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                2,
            )

            sequence_text = get_sequence_text(
                sentence_sequence
            )

            cv2.putText(
                frame,
                f"Sequence: {sequence_text}",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                2,
            )

            # =================================================
            # شريط التقدم
            # =================================================

            bar_x = 20
            bar_y = 180
            bar_width = 250
            bar_height = 18

            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (
                    bar_x + bar_width,
                    bar_y + bar_height,
                ),
                (255, 255, 255),
                2,
            )

            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (
                    bar_x
                    + int(bar_width * progress),
                    bar_y + bar_height,
                ),
                (0, 255, 0),
                -1,
            )

            # =================================================
            # عرض الترجمة عند اكتمال الجملة
            # =================================================

            if sentence_completed:
                cv2.putText(
                    frame,
                    "My name is Wejdan",
                    (20, 235),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    "Ich heisse Wejdan",
                    (20, 275),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

            cv2.putText(
                frame,
                "C: Clear | B: Backspace | Q: Quit",
                (
                    20,
                    frame.shape[0] - 20,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "Final Arabic Sign Recognition",
                frame,
            )

            # =================================================
            # أزرار التحكم
            # =================================================

            key = cv2.waitKey(1) & 0xFF

            if key == ord("c"):
                sentence_sequence.clear()

                sequence_buffer.clear()
                wrist_x_buffer.clear()

                stable_frame_count = 0
                last_motion_added_time = 0.0
                sentence_completed = False
                translation_saved = False

                clear_translation_result()

                print("Sequence cleared.")

            elif key == ord("b"):
                if sentence_sequence:
                    removed = (
                        sentence_sequence.pop()
                    )

                    print(
                        "Removed:",
                        DISPLAY_NAMES.get(
                            removed,
                            removed,
                        ),
                    )

                    sequence_buffer.clear()
                    wrist_x_buffer.clear()

                    stable_frame_count = 0
                    sentence_completed = False
                    translation_saved = False

                    clear_translation_result()

            elif key == ord("q"):
                break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()