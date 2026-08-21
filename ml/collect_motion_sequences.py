import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
HAND_MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"
MOTION_DATA_DIR = BASE_DIR / "motion_data"

CAMERA_INDEX = 0

# عدد الإطارات داخل كل عينة حركة
SEQUENCE_LENGTH = 30

# عدد العينات التي سنجمعها في كل تشغيل
TARGET_SEQUENCES = 30

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


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


def get_next_sequence_number(label_dir: Path) -> int:
    existing_files = list(label_dir.glob("sequence_*.npy"))

    if not existing_files:
        return 1

    numbers = []

    for file_path in existing_files:
        try:
            number = int(file_path.stem.split("_")[1])
            numbers.append(number)
        except (IndexError, ValueError):
            continue

    return max(numbers, default=0) + 1


def main() -> None:
    if not HAND_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"لم يتم العثور على نموذج اليد:\n{HAND_MODEL_PATH}"
        )

    label = input(
        "اكتبي اسم الحركة "
        "(esme_motion أو other_motion): "
    ).strip().lower()

    allowed_labels = {
        "esme_motion",
        "other_motion",
    }

    if label not in allowed_labels:
        raise ValueError(
            "اكتبي فقط esme_motion أو other_motion"
        )

    label_dir = MOTION_DATA_DIR / label
    label_dir.mkdir(parents=True, exist_ok=True)

    next_sequence_number = get_next_sequence_number(label_dir)

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

    recording = False
    current_sequence = []
    completed_sequences = 0

    start_time = time.perf_counter()

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

            hand_detected = bool(result.hand_landmarks)

            if hand_detected:
                landmarks = result.hand_landmarks[0]
                draw_hand(frame, landmarks)

                if recording:
                    features = normalize_landmarks(landmarks)
                    current_sequence.append(features)

                    if len(current_sequence) >= SEQUENCE_LENGTH:
                        sequence_array = np.array(
                            current_sequence,
                            dtype=np.float32,
                        )

                        file_number = (
                            next_sequence_number
                            + completed_sequences
                        )

                        save_path = (
                            label_dir
                            / f"sequence_{file_number:03d}.npy"
                        )

                        np.save(save_path, sequence_array)

                        completed_sequences += 1
                        recording = False
                        current_sequence = []

                        print(
                            f"تم حفظ العينة "
                            f"{completed_sequences}/"
                            f"{TARGET_SEQUENCES}"
                        )

            status = "RECORDING" if recording else "READY"

            cv2.putText(
                frame,
                f"Label: {label}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                (
                    f"Sequences: {completed_sequences}/"
                    f"{TARGET_SEQUENCES}"
                ),
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Status: {status}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0)
                if recording
                else (0, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                (
                    f"Frames: {len(current_sequence)}/"
                    f"{SEQUENCE_LENGTH}"
                ),
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )

            if not hand_detected:
                cv2.putText(
                    frame,
                    "Show one hand",
                    (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2,
                )

            cv2.putText(
                frame,
                "S: Record one motion | Q: Quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "Collect Motion Sequences",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if (
                key == ord("s")
                and not recording
                and completed_sequences < TARGET_SEQUENCES
            ):
                if hand_detected:
                    current_sequence = []
                    recording = True
                else:
                    print(
                        "أظهري يدك أولًا ثم اضغطي S."
                    )

            if key == ord("q"):
                break

            if completed_sequences >= TARGET_SEQUENCES:
                print("اكتمل جمع العينات.")
                break

    camera.release()
    cv2.destroyAllWindows()

    print(
        f"\nتم حفظ {completed_sequences} عينة "
        f"داخل:\n{label_dir}"
    )


if __name__ == "__main__":
    main()