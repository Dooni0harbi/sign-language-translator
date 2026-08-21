import time
from collections import deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

HAND_MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"
MOTION_MODEL_PATH = BASE_DIR / "models" / "motion_classifier.joblib"

CAMERA_INDEX = 0
SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.70

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def normalize_landmarks(landmarks):
    wrist = landmarks[0]
    features = []

    for landmark in landmarks:
        features.extend([
            landmark.x - wrist.x,
            landmark.y - wrist.y,
            landmark.z - wrist.z,
        ])

    return features


def draw_hand(frame, landmarks):
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


def main():
    if not HAND_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"لم يتم العثور على نموذج اليد:\n{HAND_MODEL_PATH}"
        )

    if not MOTION_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"لم يتم العثور على نموذج الحركة:\n{MOTION_MODEL_PATH}"
        )

    motion_model = joblib.load(MOTION_MODEL_PATH)

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

    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
    start_time = time.perf_counter()

    with mp.tasks.vision.HandLandmarker.create_from_options(
        options
    ) as detector:

        while True:
            success, frame = camera.read()

            if not success:
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

            prediction_text = "Show one hand"
            confidence_text = ""
            status_color = (0, 255, 255)

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                draw_hand(frame, landmarks)

                features = normalize_landmarks(landmarks)
                sequence_buffer.append(features)

                if len(sequence_buffer) == SEQUENCE_LENGTH:
                    sequence_array = np.array(
                        sequence_buffer,
                        dtype=np.float32,
                    ).reshape(1, -1)

                    probabilities = motion_model.predict_proba(
                        sequence_array
                    )[0]

                    best_index = int(np.argmax(probabilities))
                    confidence = float(probabilities[best_index])
                    predicted_label = motion_model.classes_[best_index]

                    confidence_text = (
                        f"Confidence: {confidence * 100:.1f}%"
                    )

                    if (
                        predicted_label == "esme_motion"
                        and confidence >= CONFIDENCE_THRESHOLD
                    ):
                        prediction_text = "ESME MOTION DETECTED"
                        status_color = (0, 255, 0)
                    else:
                        prediction_text = "OTHER MOTION"
                        status_color = (0, 165, 255)

            else:
                sequence_buffer.clear()

            cv2.putText(
                frame,
                prediction_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                status_color,
                2,
            )

            cv2.putText(
                frame,
                confidence_text,
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Frames: {len(sequence_buffer)}/{SEQUENCE_LENGTH}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                "Perform the full ESME motion",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                "R: Reset | Q: Quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "Motion Recognition",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):
                sequence_buffer.clear()

            elif key == ord("q"):
                break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()