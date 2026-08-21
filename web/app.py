import base64
import os
import threading
import time
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import serial
ARDUINO_PORT = "COM7"
ARDUINO_BAUD_RATE = 9600

# =========================================================
# المسارات
# =========================================================

WEB_DIR = Path(__file__).resolve().parent
PROJECT_DIR = WEB_DIR.parent
ML_DIR = PROJECT_DIR / "ml"

TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"
INDEX_PATH = TEMPLATES_DIR / "index.html"

HAND_MODEL_PATH = ML_DIR / "models" / "hand_landmarker.task"
STATIC_MODEL_PATH = ML_DIR / "models" / "sign_classifier.joblib"

load_dotenv(PROJECT_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_TRANSLATION_MODEL = os.getenv(
    "GEMINI_TRANSLATION_MODEL",
    "gemini-2.5-flash",
).strip()

gemini_client = (
    genai.Client(api_key=GEMINI_API_KEY)
    if GEMINI_API_KEY
    else None
)


# =========================================================
# إعدادات التعرف
# =========================================================

STATIC_CONFIDENCE_THRESHOLD = 0.80
# الإشارات الثابتة
REQUIRED_STABLE_FRAMES = 1

# منع تكرار نفس الإشارة حتى تختفي اليد قليلًا
RELEASE_FRAMES = 3


# =========================================================
# أسماء الإشارات
# =========================================================

DISPLAY_NAMES = {
    "ana": "ANA",
    "waw": "WAW",
    "jeem": "JEEM",
    "dal": "DAL",
    "alif": "ALIF",
    "noon": "NOON",
}

ARABIC_LABELS = {
    "ana": "أنا",
    "waw": "و",
    "jeem": "ج",
    "dal": "د",
    "alif": "ا",
    "noon": "ن",
}

LETTER_LABELS = {
    "waw",
    "jeem",
    "dal",
    "alif",
    "noon",
}


# =========================================================
# تحميل النماذج
# =========================================================

for required_path in (
    HAND_MODEL_PATH,
    STATIC_MODEL_PATH,
):
    if not required_path.exists():
        raise FileNotFoundError(
            f"Model file was not found: {required_path}"
        )

static_model = joblib.load(STATIC_MODEL_PATH)

base_options = mp.tasks.BaseOptions(
    model_asset_path=str(HAND_MODEL_PATH)
)

landmarker_options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)

hand_detector = (
    mp.tasks.vision.HandLandmarker.create_from_options(
        landmarker_options
    )
)


# =========================================================
# حالة الجلسة
# =========================================================

state_lock = threading.Lock()

session_state = {
    "active": False,
    "mode": None,
    "sentence_sequence": [],

    "current_candidate": None,
    "candidate_frame_count": 0,

    "locked_label": None,
    "no_hand_frames": 0,

    "last_prediction": "",
    "last_confidence": 0.0,
    "progress_type": "static",
    "status_message": "",

}


# =========================================================
# نماذج الطلبات
# =========================================================

class StartRequest(BaseModel):
    mode: str


class FrameRequest(BaseModel):
    image: str


# =========================================================
# الدوال المساعدة
# =========================================================

def normalize_label(label) -> str:
    return str(label).strip().lower()


def normalize_landmarks(landmarks) -> list[float]:
    wrist = landmarks[0]
    features: list[float] = []

    for landmark in landmarks:
        features.extend(
            [
                landmark.x - wrist.x,
                landmark.y - wrist.y,
                landmark.z - wrist.z,
            ]
        )

    return features


def decode_image(image_data: str):
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    image_bytes = base64.b64decode(image_data)
    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if frame is None:
        raise ValueError("Could not decode camera frame.")

    return frame


def get_sequence_text() -> str:
    sequence = session_state["sentence_sequence"]

    return " > ".join(
        DISPLAY_NAMES.get(label, label.upper())
        for label in sequence
    )


def build_arabic_sentence() -> str:
    sequence = session_state["sentence_sequence"]

    words: list[str] = []
    current_letters: list[str] = []

    for label in sequence:
        arabic_value = ARABIC_LABELS.get(
            label,
            label,
        )

        if label in LETTER_LABELS:
            current_letters.append(arabic_value)
            continue

        if current_letters:
            words.append("".join(current_letters))
            current_letters = []

        words.append(arabic_value)

    if current_letters:
        words.append("".join(current_letters))

    return " ".join(words).strip()


def translate_arabic_text(
    arabic_text: str,
    target_language: str,
) -> dict:
    """Translate any recognized Arabic text using Gemini API."""

    normalized_text = " ".join(arabic_text.split())

    if not normalized_text:
        return {
            "translation": "",
            "translation_available": False,
            "error": "No Arabic text was provided.",
        }

    if gemini_client is None:
        return {
            "translation": "",
            "translation_available": False,
            "error": (
                "GEMINI_API_KEY is missing. Add it to "
                "the .env file in the project root."
            ),
        }

    language_names = {
        "english": "English",
        "german": "German",
    }

    target_name = language_names.get(target_language)

    if target_name is None:
        return {
            "translation": "",
            "translation_available": False,
            "error": "Unsupported target language.",
        }

    prompt = (
        "You are a professional translator. "
        f"Translate the following Arabic text into natural {target_name}. "
        "Preserve names accurately. Return only the translation, "
        "with no explanation, labels, markdown, or quotation marks.\n\n"
        f"Arabic text: {normalized_text}"
    )

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_TRANSLATION_MODEL,
            contents=prompt,
        )

        translated_text = (
            response.text or ""
        ).strip()

        if not translated_text:
            return {
                "translation": "",
                "translation_available": False,
                "error": "Gemini returned an empty translation.",
            }

        return {
            "translation": translated_text,
            "translation_available": True,
            "error": "",
        }

    except Exception as error:
        print("Gemini translation error:", repr(error))

        return {
            "translation": "",
            "translation_available": False,
            "error": (
                "Gemini translation failed. Check the API key, "
                "internet connection, model name, and quota."
            ),
        }


def prepare_german_for_lcd(text: str) -> str:
    replacements = {
        "ä": "ae",
        "ö": "oe",
        "ü": "ue",
        "Ä": "Ae",
        "Ö": "Oe",
        "Ü": "Ue",
        "ß": "ss",
        "ẞ": "SS",
    }

    lcd_text = text

    for original, replacement in replacements.items():
        lcd_text = lcd_text.replace(
            original,
            replacement,
        )

    return " ".join(lcd_text.split()).strip()


def send_text_to_arduino(text: str) -> dict:
    lcd_text = prepare_german_for_lcd(text)

    if not lcd_text:
        return {
            "success": False,
            "error": "There is no text to send to Arduino.",
        }

    try:
        with serial.Serial(
            port=ARDUINO_PORT,
            baudrate=ARDUINO_BAUD_RATE,
            timeout=2,
            write_timeout=2,
        ) as arduino:
            time.sleep(2.0)

            arduino.reset_input_buffer()
            arduino.write(
                (lcd_text + "\n").encode("utf-8")
            )
            arduino.flush()

        print(
            f"Sent to Arduino on {ARDUINO_PORT}:",
            lcd_text,
        )

        return {
            "success": True,
            "text": lcd_text,
            "error": "",
        }

    except serial.SerialException as error:
        print("Arduino serial error:", repr(error))

        return {
            "success": False,
            "error": (
                f"Could not send text to Arduino on "
                f"{ARDUINO_PORT}. Close Arduino Serial Monitor "
                "and check the USB connection/COM port."
            ),
        }


def reset_session(mode=None) -> None:
    session_state["active"] = mode is not None
    session_state["mode"] = mode
    session_state["sentence_sequence"] = []

    session_state["current_candidate"] = None
    session_state["candidate_frame_count"] = 0


    session_state["locked_label"] = None
    session_state["no_hand_frames"] = 0

    session_state["last_prediction"] = ""
    session_state["last_confidence"] = 0.0
    session_state["progress_type"] = "static"
    session_state["status_message"] = ""



def clear_current_candidate() -> None:
    session_state["current_candidate"] = None
    session_state["candidate_frame_count"] = 0


def add_detected_label(label: str) -> None:
    normalized_label = normalize_label(label)

    session_state["sentence_sequence"].append(
        normalized_label
    )
    session_state["locked_label"] = normalized_label

    clear_current_candidate()

    session_state["progress_type"] = "static"
    session_state["status_message"] = (
        f"Recorded {DISPLAY_NAMES.get(normalized_label, normalized_label.upper())}"
    )

    print(
        "Added:",
        DISPLAY_NAMES.get(
            normalized_label,
            normalized_label.upper(),
        ),
    )
    print("Sequence:", get_sequence_text())


# =========================================================
# الإشارة الثابتة
# =========================================================

def get_static_prediction(features) -> dict:
    static_array = np.array(
        features,
        dtype=np.float32,
    ).reshape(1, -1)

    probabilities = static_model.predict_proba(
        static_array
    )[0]

    best_index = int(np.argmax(probabilities))

    return {
        "label": normalize_label(
            static_model.classes_[best_index]
        ),
        "confidence": float(
            probabilities[best_index]
        ),
    }


def update_static_candidate(
    static_result: dict,
) -> None:
    label = static_result["label"]
    confidence = static_result["confidence"]

    session_state["last_prediction"] = label
    session_state["last_confidence"] = confidence
    session_state["progress_type"] = "static"

    if confidence < STATIC_CONFIDENCE_THRESHOLD:
        clear_current_candidate()
        return

    if label == session_state["locked_label"]:
        clear_current_candidate()
        return

    if label == session_state["current_candidate"]:
        session_state["candidate_frame_count"] += 1
    else:
        session_state["current_candidate"] = label
        session_state["candidate_frame_count"] = 1

    if (
        session_state["candidate_frame_count"]
        >= REQUIRED_STABLE_FRAMES
    ):
        add_detected_label(label)


# =========================================================
# معالجة الفريم
# =========================================================

def process_frame(frame) -> dict:
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame,
    )

    result = hand_detector.detect(mp_image)

    if not result.hand_landmarks:
        session_state["no_hand_frames"] += 1

        clear_current_candidate()
            
        if (
            session_state["no_hand_frames"]
            >= RELEASE_FRAMES
        ):
            session_state["locked_label"] = None

        return build_status_response()

    session_state["no_hand_frames"] = 0

    landmarks = result.hand_landmarks[0]
    features = normalize_landmarks(landmarks)

    static_result = get_static_prediction(features)
    update_static_candidate(static_result)

    return build_status_response()


# =========================================================
# بيانات الواجهة
# =========================================================

def build_status_response() -> dict:
    current_candidate = session_state[
        "current_candidate"
    ]

    if current_candidate is not None:
        progress = session_state[
            "candidate_frame_count"
        ]
    else:
        progress = 0

    return {
        "active": session_state["active"],
        "mode": session_state["mode"],
        "completed": bool(
            session_state["sentence_sequence"]
        ),
        "sequence": session_state[
            "sentence_sequence"
        ],
        "sequence_text": get_sequence_text(),
        "arabic_preview": build_arabic_sentence(),
        "expected": "Any trained static sign",
        "prediction": session_state[
            "last_prediction"
        ],
        "confidence": round(
            session_state[
                "last_confidence"
            ] * 100,
            1,
        ),
        "progress": progress,
        "required_frames": REQUIRED_STABLE_FRAMES,
        "progress_type": "static",
        "current_candidate": current_candidate,
        "locked_label": session_state[
            "locked_label"
        ],
        "status_message": session_state[
            "status_message"
        ],
    }


# =========================================================
# FastAPI
# =========================================================

app = FastAPI(
    title="Arabic Sign Language Translator"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://arabic-translator-travel-camera.netlify.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)


@app.get("/", response_class=HTMLResponse)
def home():
    if not INDEX_PATH.exists():
        return HTMLResponse(
            "index.html was not found.",
            status_code=404,
        )

    return HTMLResponse(
        INDEX_PATH.read_text(
            encoding="utf-8"
        )
    )


@app.get("/api/status")
def status():
    with state_lock:
        return build_status_response()


@app.post("/api/start")
def start_translation(
    request: StartRequest,
):
    mode = request.mode.strip().lower()

    if mode not in {"english", "german"}:
        return {
            "success": False,
            "error": (
                "Mode must be english or german."
            ),
        }

    with state_lock:
        reset_session(mode)

        return {
            "success": True,
            "mode": mode,
            "message": (
                "Camera recognition started."
            ),
        }


@app.post("/api/frame")
def receive_frame(
    request: FrameRequest,
):
    with state_lock:
        if not session_state["active"]:
            return {
                "success": False,
                "error": (
                    "No active translation session."
                ),
            }

        try:
            frame = decode_image(request.image)
            result = process_frame(frame)

            return {
                "success": True,
                **result,
            }

        except (
            ValueError,
            TypeError,
            base64.binascii.Error,
        ) as error:
            return {
                "success": False,
                "error": str(error),
            }


@app.post("/api/stop")
def stop_translation():
    with state_lock:
        mode = session_state["mode"]
        sequence = list(
            session_state["sentence_sequence"]
        )
        arabic_text = build_arabic_sentence()
        sequence_text = get_sequence_text()
        session_state["active"] = False

    result = {
        "success": True,
        "mode": mode,
        "completed": bool(sequence),
        "sequence": sequence,
        "sequence_text": sequence_text,
        "arabic": arabic_text,
        "english": "",
        "german": "",
        "translation_available": False,
    }

    if not sequence or not arabic_text:
        result["message"] = "No signs were recognized."
        return result

    translation_result = translate_arabic_text(
        arabic_text,
        mode,
    )

    result["translation_available"] = (
        translation_result["translation_available"]
    )

    if not translation_result["translation_available"]:
        result["message"] = translation_result["error"]
        return result

    translated_text = translation_result["translation"]

    if mode == "english":
        result["english"] = translated_text
        result["action"] = "speak_english"
    elif mode == "german":
        result["german"] = translated_text
        result["action"] = "send_to_arduino"

        arduino_result = send_text_to_arduino(
            translated_text
        )

        result["arduino_sent"] = arduino_result[
            "success"
        ]

        if arduino_result["success"]:
            result["arduino_text"] = arduino_result[
                "text"
            ]
            result["message"] = (
                "German translation was sent to the LCD."
            )
        else:
            result["arduino_error"] = arduino_result[
                "error"
            ]
            result["message"] = arduino_result[
                "error"
            ]

    return result


@app.post("/api/backspace")
def remove_last_sign():
    with state_lock:
        sequence = session_state[
            "sentence_sequence"
        ]

        if not sequence:
            return {
                "success": False,
                "message": (
                    "There are no signs to remove."
                ),
            }

        removed = sequence.pop()
        session_state["locked_label"] = None
        clear_current_candidate()

        return {
            "success": True,
            "removed": removed,
            "sequence_text": get_sequence_text(),
            "arabic_preview": build_arabic_sentence(),
        }


@app.post("/api/clear")
def clear_signs():
    with state_lock:
        current_mode = session_state["mode"]
        reset_session(current_mode)

        return {
            "success": True,
            "message": (
                "Recorded signs were cleared."
            ),
        } 