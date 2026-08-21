import csv
import time
from pathlib import Path

import cv2
import mediapipe as mp


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"
DATA_DIR = BASE_DIR / "data"
CSV_PATH = DATA_DIR / "landmarks.csv"

CAMERA_INDEX = 0
TARGET_SAMPLES = 50
CURRENT_LABEL = "ana "  # حرف و


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def draw_hand(frame, landmarks) -> None:
    height, width, _ = frame.shape
    points = []

    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((x, y))

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (255, 255, 255), 2)

    for point in points:
        cv2.circle(frame, point, 5, (0, 255, 0), -1)


def normalize_landmarks(landmarks) -> list[float]:
    """
    نحول النقاط إلى قيم نسبية مقارنة بنقطة الرسغ.
    هذا يقلل تأثير مكان اليد داخل الشاشة.
    """
    wrist = landmarks[0]

    normalized = []

    for landmark in landmarks:
        normalized.extend([
            landmark.x - wrist.x,
            landmark.y - wrist.y,
            landmark.z - wrist.z,
        ])

    return normalized


def create_csv_header() -> list[str]:
    header = ["label"]

    for index in range(21):
        header.extend([
            f"x{index}",
            f"y{index}",
            f"z{index}",
        ])

    return header


def save_sample(label: str, features: list[float]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    file_exists = CSV_PATH.exists()

    with CSV_PATH.open(
        mode="a",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.writer(csv_file)

        if not file_exists:
            writer.writerow(create_csv_header())

        writer.writerow([label, *features])


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    base_options = mp.tasks.BaseOptions(
        model_asset_path=str(MODEL_PATH)
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

    collecting = False
    sample_count = 0
    last_save_time = 0.0
    save_interval = 0.12

    start_time = time.perf_counter()

    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
        while True:
            success, frame = camera.read()

            if not success:
                print("تعذر قراءة الصورة من الكاميرا.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

            hand_detected = bool(result.hand_landmarks)

            if hand_detected:
                hand_landmarks = result.hand_landmarks[0]
                draw_hand(frame, hand_landmarks)

                if collecting and sample_count < TARGET_SAMPLES:
                    current_time = time.perf_counter()

                    if current_time - last_save_time >= save_interval:
                        features = normalize_landmarks(hand_landmarks)
                        save_sample(CURRENT_LABEL, features)

                        sample_count += 1
                        last_save_time = current_time

                if sample_count >= TARGET_SAMPLES:
                    collecting = False

            status_text = (
                "COLLECTING"
                if collecting
                else "READY"
            )

            cv2.putText(
                frame,
                f"Label: {CURRENT_LABEL}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Samples: {sample_count}/{TARGET_SAMPLES}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Status: {status_text}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if collecting else (0, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                "Press S to start | Q to quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            if not hand_detected:
                cv2.putText(
                    frame,
                    "Show one hand",
                    (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            if sample_count >= TARGET_SAMPLES:
                cv2.putText(
                    frame,
                    "Collection complete",
                    (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow(
                "Collect Hand Landmarks",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s") and sample_count < TARGET_SAMPLES:
                collecting = True

            if key == ord("q"):
                break

    camera.release()
    cv2.destroyAllWindows()

    print(f"Saved samples: {sample_count}")
    print(f"CSV file: {CSV_PATH}")


if __name__ == "__main__":
    main()