# Travel Sign Translator

<p align="center">
  <strong>AI-Powered Arabic Sign Language Recognition, Multilingual Translation, Voice Output, and Arduino LCD Integration</strong>
</p>

<p align="center">
  Arabic Sign Language → AI Recognition → Arabic Text → English Voice / German LCD
</p>

---

## Overview

**Travel Sign Translator** is an end-to-end AI project that combines **Computer Vision**, **Machine Learning**, **Generative AI**, **Full-Stack Web Development**, and **Embedded Systems**.

The system recognizes a limited set of Arabic Sign Language gestures through a live camera, converts them into Arabic text, translates the text using **Google Gemini**, and delivers the result in one of two ways:

- **English Voice:** the translated sentence is spoken by the browser.
- **German LCD:** the translated sentence is sent through **Web Serial** to an **Arduino Uno** and displayed on a **16x2 I2C LCD**.

> The production backend is hosted separately. Runtime secrets are stored only in secure server-side environment variables and are never exposed in the frontend or README.

### Live Application

**Frontend:**  
https://arabic-translator-travel-camera.netlify.app

**Repository:**  
https://github.com/Dooni0harbi/sign-language-translator

---

## Demo

A project demo can show the complete workflow:

1. Perform Arabic Sign Language gestures in front of the camera.
2. Detect hand landmarks in real time.
3. Classify the gesture using the trained ML model.
4. Build Arabic text.
5. Translate with Gemini.
6. Produce English voice output or German text.
7. Send German text to Arduino using Web Serial.
8. Display the translated text on the physical LCD.

### Demo Video


https://github.com/user-attachments/assets/6a7caca0-e74f-4460-80bb-696c4b816bf3

---

## Screenshots

### Main Interface
<img width="1883" height="909" alt="لقطة شاشة 2026-08-10 235100" src="https://github.com/user-attachments/assets/eec416be-d2fc-4d02-abe1-7ead3b7d04a4" />
<img width="1836" height="898" alt="لقطة شاشة 2026-08-10 235123" src="https://github.com/user-attachments/assets/e0dfccc1-b852-45df-8625-974e1afaf7e4" />

### Real-Time Sign Recognition
<img width="1080" height="1350" alt="Untitled design" src="https://github.com/user-attachments/assets/8a1733c3-6303-41bc-93e7-a4977c01fe83" />


### Translation Result

<img width="1284" height="2778" alt="IMG_5165" src="https://github.com/user-attachments/assets/77c76524-c996-4290-a47f-7007e925c9a6" />

### Arduino LCD Integration

<img width="1284" height="2778" alt="IMG_5166" src="https://github.com/user-attachments/assets/90271490-cd55-4b4c-a516-5e4ddf01220f" />


---

## System Architecture

```text
Human Hand
    ↓
Browser Camera
    ↓
React / Vite Frontend
    ↓
JPEG Frame via REST API
    ↓
FastAPI Backend
    ↓
OpenCV Image Decode
    ↓
MediaPipe Hand Landmarker
    ↓
21 Hand Landmarks
    ↓
63 Normalized Features
    ↓
Scikit-learn Classifier
    ↓
Arabic Sign Sequence
    ↓
Arabic Sentence
    ↓
Google Gemini
    ├──→ English Translation → Browser Speech
    └──→ German Translation  → Web Serial → Arduino Uno → LCD
```

---

## Main Features

- Live webcam capture from the browser
- MediaPipe hand detection
- 21-point hand landmark extraction
- Static gesture classification
- Confidence filtering
- Duplicate-sign prevention
- Arabic sentence construction
- Arabic-to-English translation
- Arabic-to-German translation
- English browser speech output
- German Arduino LCD output
- Web Serial communication
- React/Vite frontend
- FastAPI backend
- Netlify frontend deployment
- Render backend deployment
- Secure Gemini environment variables
- CORS configuration
- Git/GitHub version control

---

## Technology Stack

| Area | Technology |
|---|---|
| Machine Learning | scikit-learn |
| Hand Tracking | MediaPipe Tasks Vision |
| Computer Vision | OpenCV |
| Numerical Processing | NumPy |
| Model Serialization | Joblib |
| Backend | FastAPI |
| API Server | Uvicorn |
| AI Translation | Google Gemini API |
| Gemini SDK | `google-genai` |
| Frontend | React |
| Build Tool | Vite |
| UI Animation | Framer Motion |
| Icons | Lucide React |
| English Voice | Web Speech API / `speechSynthesis` |
| Hardware Communication | Web Serial API |
| Microcontroller | Arduino Uno |
| Display | SunFounder 16x2 I2C LCD / LCD1602 |
| Arduino LCD Library | `LiquidCrystal_I2C` |
| Frontend Hosting | Netlify |
| Backend Hosting | Render |
| Version Control | Git + GitHub |
| Containerization | Not currently used; Docker is a future improvement |

---

# 1. Programming Languages

## Python

Python is used for:

- machine-learning inference
- MediaPipe integration
- OpenCV image processing
- FastAPI backend
- Gemini API integration
- model loading
- recognition-session state management

## JavaScript / JSX

JavaScript and JSX are used for:

- React UI
- browser camera access
- REST API requests
- frontend state management
- Web Serial communication
- English speech synthesis
- user interaction logic

## HTML / CSS

Used through React/Vite for:

- page structure
- responsive layout
- camera view
- cards
- buttons
- status indicators
- result presentation

## Arduino C/C++

Used for:

- USB serial input
- LCD initialization
- receiving German text
- splitting text across two LCD lines

---

# 2. Machine Learning Development

## 2.1 Why Hand Landmarks Instead of Raw Images?

The project uses **MediaPipe Hand Landmarker** to convert each detected hand into structured geometric information instead of training directly on the complete camera image.

MediaPipe returns **21 landmarks** for one detected hand.

Each landmark contains:

```text
x, y, z
```

Therefore:

```text
21 landmarks × 3 coordinates = 63 numerical features
```

### Advantages

- smaller input representation than full images
- faster inference
- reduced sensitivity to background
- easier debugging
- no GPU requirement for the classifier
- suitable for real-time webcam recognition

---

## 2.2 Final Static Sign Classes

The final static version uses these labels:

| Model Label | Arabic Output | Type |
|---|---|---|
| `ana` | أنا | Static word/sign |
| `waw` | و | Static letter |
| `jeem` | ج | Static letter |
| `dal` | د | Static letter |
| `alif` | ا | Static letter |
| `noon` | ن | Static letter |

The letters can be combined into a word:

```text
waw + jeem + dal + alif + noon
```

Result:

```text
وجدان
```

---

## 2.3 Landmark Normalization

The classifier does not use the absolute landmark coordinates directly.

The wrist landmark, index `0`, is used as a local origin.

For each landmark:

```python
normalized_x = landmark.x - wrist.x
normalized_y = landmark.y - wrist.y
normalized_z = landmark.z - wrist.z
```

The final feature vector becomes:

```text
[x0, y0, z0, x1, y1, z1, ... x20, y20, z20]
```

Total:

```text
63 features
```

This makes the classifier less sensitive to where the hand appears inside the camera frame.

---

## 2.4 Static Classifier

The trained model is stored as:

```text
ml/models/sign_classifier.joblib
```

It is loaded with:

```python
static_model = joblib.load(STATIC_MODEL_PATH)
```

The deployed model is a **scikit-learn RandomForestClassifier**.

The backend uses:

```python
predict_proba(...)
```

so it can obtain both:

- predicted class
- confidence probability

Inference flow:

```text
Camera Frame
→ MediaPipe
→ 21 Landmarks
→ 63 Normalized Features
→ Random Forest
→ Class Probabilities
→ Confidence Threshold
→ Accept / Reject Sign
```

---

## 2.5 Training Workflow

The runtime project contains the trained model and the inference feature format. The original first dataset-generation script and exact training counts are not present in the available artifacts, so this documentation does not invent values that were not recorded.

The training workflow is:

1. Define one class label for each gesture.
2. Capture multiple examples of each sign using a webcam.
3. Detect one hand with MediaPipe Hand Landmarker.
4. Extract 21 hand landmarks.
5. Normalize landmarks relative to the wrist.
6. Flatten the landmarks into 63 features.
7. Store features with the correct gesture label.
8. Train a `RandomForestClassifier`.
9. Test recognition using unseen or live poses.
10. Export the trained model with Joblib.
11. Load the model inside FastAPI for real-time inference.

Conceptual dataset row:

```text
x0,y0,z0,x1,y1,z1,...,x20,y20,z20,label
```

Useful data variation includes:

- hand position
- small rotations
- distance from camera
- lighting
- natural finger-placement variation

---

## 2.6 ML and Computer Vision Libraries

```text
mediapipe
opencv-python
opencv-python-headless
numpy
scikit-learn
joblib
```

Example installation:

```powershell
pip install mediapipe
pip install opencv-python
pip install numpy
pip install scikit-learn
pip install joblib
```

For cloud deployment, `opencv-python-headless` is preferable because the backend does not need desktop GUI windows.

---

## 2.7 Confidence Threshold

The project initially used lower confidence thresholds. During testing, visually similar signs could be confused.

The static threshold was therefore increased during tuning to approximately:

```python
STATIC_CONFIDENCE_THRESHOLD = 0.85
```

Logic:

```text
confidence < threshold  → ignore
confidence >= threshold → accept as candidate
```

A higher threshold can reduce false positives, but it may also reject some correct gestures.

---

## 2.8 Duplicate Prevention

A camera sends many frames while the user is holding the same pose.

Without duplicate protection, one sign could be recorded repeatedly.

The backend therefore tracks state such as:

```text
current_candidate
candidate_frame_count
locked_label
no_hand_frames
```

Conceptually:

```text
Recognize Sign
→ Record Once
→ Lock Same Label
→ Ignore Repeated Frames
→ Hand Disappears or Changes
→ Release Lock
→ Allow Next Sign
```

---

# 3. Historical Motion Recognition Phase

An earlier version also experimented with the dynamic word:

```text
اسمي
```

using:

```text
esme_motion
```

That development phase used:

```text
motion_classifier.joblib
```

and temporal buffers of approximately 30 frames.

Motion-related parameters included:

```text
SEQUENCE_LENGTH
MOTION_CONFIDENCE_THRESHOLD
MIN_HORIZONTAL_DISPLACEMENT
MIN_HORIZONTAL_RANGE
MOTION_COOLDOWN
```

The system tracked wrist movement over time to distinguish the motion sign from static gestures.

Later, `اسمي` was removed from the required vocabulary. Since it was the only motion sign, the motion-specific model and logic were removed from the final application.

The final recognition pipeline uses:

```text
hand_landmarker.task
sign_classifier.joblib
```

This reduced complexity and improved maintainability.

---

# 4. Real-Time Browser Recognition Pipeline

The browser does not stream the raw webcam directly to FastAPI.

React captures a reduced frame using a hidden canvas.

Current frontend values:

```text
Frame size: 256 × 192
Encoding: JPEG
JPEG quality: 0.55
Frame interval: approximately 40 ms
```

The canvas mirrors the image:

```javascript
context.translate(width, 0);
context.scale(-1, 1);
```

Then the frame is converted to Base64 JPEG:

```javascript
canvas.toDataURL("image/jpeg", 0.55)
```

and sent to:

```text
POST /api/frame
```

---

# 5. Backend Frame Processing

FastAPI processes each frame through this pipeline:

```text
Base64 JPEG
→ Base64 Decode
→ NumPy Buffer
→ OpenCV imdecode
→ BGR to RGB
→ MediaPipe Image
→ Hand Landmarker
→ 21 Landmarks
→ Wrist-Relative Normalization
→ Random Forest Probabilities
→ Confidence Threshold
→ Session Sequence
```

---

# 6. Arabic Sentence Construction

Recognized labels are stored inside a session sequence.

Example mapping:

```python
ARABIC_LABELS = {
    "ana": "أنا",
    "waw": "و",
    "jeem": "ج",
    "dal": "د",
    "alif": "ا",
    "noon": "ن",
}
```

Letter labels are concatenated into one word.

Example sequence:

```text
["ana", "waw", "jeem", "dal", "alif", "noon"]
```

Conceptual Arabic output:

```text
أنا وجدان
```

When the user presses **إيقاف وترجمة**, FastAPI builds the final Arabic text and sends it to Gemini.

---

# 7. Gemini API Integration

The translation layer uses the official Google GenAI SDK:

```python
from google import genai
```

Installed with:

```powershell
pip install google-genai
```

Configured model:

```text
gemini-2.5-flash
```

Environment variables:

```text
GEMINI_API_KEY
GEMINI_TRANSLATION_MODEL
```

The prompt is designed to:

- translate Arabic naturally
- preserve names
- return only the translation
- avoid explanations
- avoid markdown
- avoid quotation marks

> The Gemini API key is stored on the backend only. It is never stored in the React frontend, README, or committed source files.

---

# 8. English Voice Output

When **English Voice** is selected, Gemini returns English text.

The frontend uses:

```javascript
new SpeechSynthesisUtterance(text)
```

Typical configuration:

```text
Language: en-US
Rate: 0.9
Pitch: 1
```

The speech is generated by the browser, not by FastAPI.

---

# 9. Why I Added German 🇩🇪

German was intentionally selected as one of the target languages for both technical and personal reasons.

I am personally interested in the German language and I am currently learning it. Because of that, I wanted the project to include German instead of limiting the translation workflow to English only.

I also wanted the German mode to demonstrate something technically different. Instead of only displaying translated text inside the browser, I connected the web application to an **Arduino Uno** and a physical **16x2 LCD**.

The German workflow is:

```text
Arabic Sign Language
→ Computer Vision
→ Machine Learning
→ Arabic Text
→ Gemini German Translation
→ Web Serial
→ Arduino Uno
→ Physical LCD Display
```

This makes the German feature both personally meaningful and technically useful. It demonstrates how an AI-powered web application can communicate with real embedded hardware.

---

# 10. German LCD Output

When **German LCD** is selected:

```text
Arabic Signs
→ Arabic Sentence
→ Gemini German Translation
→ German Text Shown in React
→ User Presses "إرسال إلى الشاشة"
→ Web Serial
→ Arduino Uno
→ LCD
```

The final UX intentionally does **not** auto-send the German translation.

The translation is shown first, then the user explicitly decides when to send it to the physical display.

---

# 11. Arduino Hardware

Hardware used:

- Arduino Uno
- SunFounder 16x2 I2C LCD / LCD1602
- USB cable
- jumper wires

A breadboard is not required because the LCD uses an I2C backpack.

---

# 12. LCD Wiring

| LCD I2C Pin | Arduino Uno |
|---|---|
| GND | GND |
| VCC | 5V |
| SDA | A4 |
| SCL | A5 |

Diagram:

```text
LCD                 Arduino Uno
GND  -------------- GND
VCC  -------------- 5V
SDA  -------------- A4
SCL  -------------- A5
```

Tested I2C address:

```text
0x27
```

---

# 13. Arduino Library

Install from Arduino IDE:

```text
Library Manager
→ Search: LiquidCrystal_I2C
→ Install
```

The firmware uses:

```cpp
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
```

---

# 14. Arduino Firmware

```cpp
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x27, 16, 2);

String incomingText = "";

void setup() {
  Serial.begin(9600);

  lcd.init();
  lcd.backlight();
  lcd.clear();

  lcd.setCursor(0, 0);
  lcd.print("German LCD");

  lcd.setCursor(0, 1);
  lcd.print("Ready...");
}

void loop() {
  if (Serial.available() > 0) {
    incomingText = Serial.readStringUntil('\n');
    incomingText.trim();

    if (incomingText.length() > 0) {
      lcd.clear();

      lcd.setCursor(0, 0);
      lcd.print(incomingText.substring(0, 16));

      if (incomingText.length() > 16) {
        lcd.setCursor(0, 1);
        lcd.print(incomingText.substring(16, 32));
      }
    }
  }
}
```

Serial configuration:

```text
9600 baud
```

The LCD provides two rows of 16 characters.

---

# 15. Serial Architecture Evolution

## Initial Local Version: Python `pyserial`

The first local hardware integration opened a Windows serial port such as:

```text
COM7
```

using:

```python
import serial
```

Installed with:

```powershell
pip install pyserial
```

This works when FastAPI and Arduino are physically connected to the same computer.

## Why It Could Not Stay in the Cloud Backend

After deploying FastAPI to Render, the cloud server could not access the laptop's local `COM7` port.

This architecture is not possible directly:

```text
Render Server → User Laptop COM7
```

without an additional local bridge.

## Final Solution: Web Serial

Serial communication was moved to the browser:

```text
Netlify Website
→ Browser
→ Web Serial API
→ Local USB
→ Arduino Uno
→ LCD
```

The browser requests permission using:

```javascript
navigator.serial.requestPort()
```

This is the correct hosted architecture because the browser is running on the same physical computer as the Arduino.

---

# 16. German Character Conversion

Basic HD44780-compatible LCDs may not display UTF-8 German characters correctly.

Before serial transmission, the frontend converts:

```text
ä → ae
ö → oe
ü → ue
Ä → Ae
Ö → Oe
Ü → Ue
ß → ss
```

---

# 17. Web Serial User Flow

When **German LCD** is selected:

1. React checks whether a serial port is already connected.
2. If not, it displays a message explaining that an Arduino port must be selected.
3. The browser opens the native serial-port chooser.
4. The user selects the Arduino.
5. Translation continues normally.
6. German text is sent only when the user presses **إرسال إلى الشاشة**.

> Arduino Serial Monitor must be closed while the browser owns the serial port.

---

# 18. Frontend Architecture

Main frontend file:

```text
frontend/src/App.jsx
```

Responsibilities:

- camera permission
- video stream
- hidden canvas capture
- API requests
- recognition status
- confidence display
- Arabic preview
- translation output
- English speech
- Web Serial connection
- LCD send action

Important React state includes:

```text
isRunning
selectedLanguage
prediction
confidence
arabicText
translatedText
progress
requiredFrames
statusMessage
isArduinoConnected
```

Important refs include:

```text
videoRef
canvasRef
streamRef
timerRef
sendingFrameRef
serialPortRef
serialWriterRef
```

---

# 19. Backend Architecture

Main backend file:

```text
web/app.py
```

Responsibilities:

- load `hand_landmarker.task`
- load `sign_classifier.joblib`
- decode incoming frames
- extract hand landmarks
- normalize features
- run classifier probabilities
- manage recognition state
- build Arabic text
- call Gemini
- return JSON responses

---

# 20. API Endpoints

## `GET /api/status`

Returns the current recognition state.

## `POST /api/start`

Starts a new translation session.

Example:

```json
{
  "mode": "english"
}
```

or:

```json
{
  "mode": "german"
}
```

## `POST /api/frame`

Receives a Base64 JPEG frame.

Example:

```json
{
  "image": "data:image/jpeg;base64,..."
}
```

## `POST /api/stop`

Stops recognition, builds Arabic text, calls Gemini, and returns the translation.

Development versions also included:

```text
/api/backspace
/api/clear
```

---

# 21. CORS

The frontend and backend are deployed on separate domains.

```text
Netlify → Frontend
Render  → Backend
```

FastAPI therefore uses CORS middleware:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://arabic-translator-travel-camera.netlify.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Without the correct CORS configuration, the browser may show:

```text
Failed to fetch
```

---

# 22. Project Structure

```text
sign-language-translator/
│
├── frontend/
│   ├── src/
│   │   └── App.jsx
│   ├── package.json
│   └── ...
│
├── web/
│   ├── app.py
│   ├── templates/
│   └── static/
│
├── ml/
│   └── models/
│       ├── hand_landmarker.task
│       └── sign_classifier.joblib
│
├── requirements.txt
├── .gitignore
└── README.md
```

Historical development versions also contained:

```text
motion_classifier.joblib
```

---

# 23. Python Dependencies

Typical project requirements:

```txt
fastapi
uvicorn
opencv-python-headless
mediapipe
numpy
joblib
scikit-learn
python-dotenv
google-genai
pyserial
```

> `pyserial` belongs to the earlier local serial-development phase. The final hosted Arduino communication path uses Web Serial in the frontend.

---

# 24. Local Installation

Clone the repository:

```powershell
git clone https://github.com/Dooni0harbi/sign-language-translator.git
cd sign-language-translator
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\activate
```

Install Python requirements:

```powershell
pip install -r requirements.txt
```

Install frontend packages:

```powershell
cd frontend
npm install
```

---

# 25. Local Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=YOUR_REAL_API_KEY
GEMINI_TRANSLATION_MODEL=gemini-2.5-flash
```

> Never commit the real `.env` file or API key to GitHub.

---

# 26. Running Locally

## Backend

```powershell
cd C:\Users\DoOod\OneDrive\Desktop\sign-language-translator
.\.venv\Scripts\activate
uvicorn web.app:app --reload
```

Local backend:

```text
http://127.0.0.1:8000
```

## Frontend

```powershell
cd C:\Users\DoOod\OneDrive\Desktop\sign-language-translator\frontend
npm run dev
```

Local frontend:

```text
http://localhost:5173
```

Optional LAN testing:

```powershell
npm run dev -- --host
```

---

# 27. Production Backend Configuration

The production frontend communicates with a separately hosted FastAPI service.

The public README intentionally does not expose the production backend URL directly.

For a cleaner deployment architecture, the frontend can read the API base URL from a deployment variable:

```javascript
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;
```

Example requests:

```javascript
fetch(`${API_BASE_URL}/api/start`, ...)
fetch(`${API_BASE_URL}/api/frame`, ...)
fetch(`${API_BASE_URL}/api/stop`, ...)
```

Important distinction:

```text
Backend address     = application infrastructure
Gemini API key      = secret
Environment values  = server-side configuration
```

The Gemini key must remain exclusively in server-side environment variables.

---

# 28. Render Backend Deployment

Service type:

```text
Web Service
```

Repository:

```text
Dooni0harbi/sign-language-translator
```

Branch:

```text
main
```

Build command:

```text
pip install -r requirements.txt
```

Start command:

```text
uvicorn web.app:app --host 0.0.0.0 --port $PORT
```

Environment variables:

```text
GEMINI_API_KEY=<stored securely in Render>
GEMINI_TRANSLATION_MODEL=gemini-2.5-flash
```

The real key is never committed to GitHub and is never placed in frontend code.

---

# 29. Netlify Frontend Deployment

Branch:

```text
main
```

Base directory:

```text
frontend
```

Build command:

```text
npm run build
```

Publish directory:

```text
dist
```

Live application:

```text
https://arabic-translator-travel-camera.netlify.app
```

---

# 30. Security Design

Correct architecture:

```text
React → FastAPI → Gemini
```

Production key location:

```text
Render → Environment → GEMINI_API_KEY
```

Local key location:

```text
.env
```

Security rules:

- never place the Gemini API key in `App.jsx`
- never commit `.env`
- never publish the API key in README
- keep CORS restricted to the frontend domain
- use server-side environment variables for secrets
- add rate limiting and abuse protection in a future public production version

---

# 31. Git / GitHub Workflow

Initial setup:

```powershell
git init
git add .
git commit -m "first commit"
git remote add origin https://github.com/Dooni0harbi/sign-language-translator.git
git branch -M main
git push -u origin main
```

Normal update workflow:

```powershell
git add .
git commit -m "describe the change"
git push
```

If the remote contains newer changes:

```powershell
git pull --rebase origin main
git push
```

---

# 32. Important Engineering Decisions

### Landmark features instead of raw image classification

Reduced computational load and made real-time inference practical.

### Wrist-relative normalization

Reduced dependence on absolute hand position inside the camera frame.

### Confidence threshold

Prevented the application from trusting every model prediction.

### Sign locking

Prevented one held pose from being recorded repeatedly.

### Removing unnecessary motion recognition

The motion subsystem was removed when the final vocabulary no longer required the dynamic `اسمي` sign.

### Gemini on FastAPI, not React

Protected the API key from browser inspection.

### Web Serial instead of cloud `pyserial`

Solved the architectural problem that a Render server cannot directly access a USB port on the user's laptop.

### Manual German LCD send

The user sees the translation first and explicitly decides when to send it to the hardware.

---

# 33. Debugging History

## Wrong Arduino Board

Arduino IDE was initially configured for an ESP32/Wemos board instead of Arduino Uno.

Selecting **Arduino Uno** fixed the upload path.

## COM7 Access Denied

If Web Serial or Arduino Serial Monitor owns COM7, firmware upload may fail with:

```text
Access is denied
```

Close the competing serial connection first.

## LCD Test

A working LCD test displayed:

```text
German LCD
Ready...
```

before web integration.

## `Unexpected end of JSON input`

This can happen when the frontend attempts to parse a response while the backend is not available or does not return the expected JSON.

## `Failed to fetch`

After hosting frontend and backend separately, this was solved using the correct backend API base URL and CORS configuration.

## scikit-learn Warning

Render may log:

```text
X does not have valid feature names,
but RandomForestClassifier was fitted with feature names
```

This warning occurs because the model was trained using named feature columns while inference passes a NumPy array.

It does not automatically mean prediction failed.

## Gemini Quota

Gemini Free Tier can return:

```text
429 TooManyRequests
```

A new API key inside the same Google project normally shares the same project-level quota.

---

# 34. Current Limitations

- prototype vocabulary only
- not a complete Arabic Sign Language system
- one-hand recognition
- final version focuses on static gestures
- similar poses may still be confused
- performance depends on dataset diversity
- no continuous sign-language segmentation
- no Arabic language-model correction before translation
- LCD is limited to 16x2 characters at one time
- Web Serial support is browser-dependent
- Gemini free quota can interrupt translation
- hosted free services may have startup delays
- broader multi-user evaluation is required before accessibility or production use

---

# 35. Future Development

The current system is a functional prototype, but it can be expanded into a significantly more capable assistive communication platform.

## 35.1 Expand the Arabic Sign Language Dataset

Future versions can include:

- complete Arabic alphabet
- common words
- daily expressions
- travel phrases
- emergency phrases
- healthcare communication
- numbers
- questions
- greetings
- two-handed signs
- samples from multiple signers

The dataset should also contain greater variation in:

- hand size
- skin tone
- camera angle
- distance
- background
- lighting
- signing speed

---

## 35.2 Continuous Sign Language Recognition

A more advanced version could recognize continuous signing without requiring the user to separate every sign manually.

```text
Live Video
→ Hand / Pose / Face Tracking
→ Temporal Segmentation
→ Sign Sequence Model
→ Arabic Sentence Reconstruction
→ Translation
```

Possible methods:

- LSTM
- GRU
- Temporal Convolutional Networks
- Transformers
- sequence-to-sequence models

---

## 35.3 Reintroduce Dynamic Gesture Recognition

An earlier version experimented with the motion sign `اسمي`.

A future implementation could use complete landmark sequences:

```text
30–60 Landmark Frames
→ Sequence Normalization
→ LSTM / GRU / Transformer
→ Dynamic Sign Class
```

---

## 35.4 Add Two-Hand, Body, and Facial Features

Future models could combine:

- both hands
- arm movement
- body pose
- facial expressions
- head orientation

MediaPipe Hand Landmarker, Pose Landmarker, and Face Landmarker could be combined into a multimodal recognition pipeline.

---

## 35.5 Improve Model Evaluation

Future ML work should include:

- train/validation/test splits
- confusion matrices
- precision
- recall
- F1-score
- cross-validation
- class imbalance analysis
- per-class error analysis
- hyperparameter tuning

Models worth comparing:

- Random Forest
- SVM
- XGBoost
- MLP
- LSTM
- GRU
- Transformers

---

## 35.6 Personalized Calibration

A future version could allow users to record several examples of their own signing style and calibrate the recognizer without retraining the complete system.

---

## 35.7 Add More Spoken Languages

Because translation is separate from gesture recognition, additional languages can be added without retraining the sign classifier.

Possible future languages:

- French
- Spanish
- Italian
- Turkish
- Japanese
- Korean

German will remain an important language in the project because I am personally interested in the language and currently learning it.

---

## 35.8 Expand the German Learning Experience

Future German features could include:

- German text-to-speech
- pronunciation guidance
- vocabulary explanations
- formal vs. informal phrasing
- travel expressions
- translation history
- pronunciation practice
- offline phrase packs

This could make the system useful as both an accessibility experiment and a language-learning interface.

---

## 35.9 Improve the Physical Display

The 16x2 LCD could be replaced or extended with:

- larger LCD
- OLED
- TFT display
- touchscreen
- e-paper display
- scrolling text

---

## 35.10 Wireless Hardware Communication

Future hardware could use:

- Bluetooth
- BLE
- Wi-Fi
- ESP32
- Web Bluetooth
- WebSockets
- MQTT

This would remove the need for a permanent USB serial cable.

---

## 35.11 Portable Standalone Device

A future version could run on:

- Raspberry Pi
- NVIDIA Jetson
- Edge AI accelerator
- ESP32 for peripheral control

Possible architecture:

```text
Camera
→ Edge AI Device
→ Sign Recognition
→ Translation
→ Speaker / Display
```

---

## 35.12 Offline Translation

A local translation model could reduce dependence on:

- internet access
- API quotas
- external cloud availability

Potential benefits:

- privacy
- lower latency
- resilience
- offline use

---

## 35.13 Arabic Language Correction Before Translation

A future NLP layer could reconstruct and correct the Arabic sentence before translation:

```text
Raw Sign Predictions
→ Arabic Sentence Reconstruction
→ Grammar / Language Correction
→ Translation
```

---

## 35.14 Translation and Session History

The frontend could save:

- recognized Arabic text
- selected target language
- translated output
- timestamp
- confidence
- voice replay
- LCD resend action

---

## 35.15 User Accounts and Preferences

Optional profiles could store:

- preferred language
- accessibility settings
- speech speed
- favorite phrases
- personalized calibration
- device preferences
- history

---

## 35.16 Mobile Application

A future Android or iOS app could integrate:

- camera
- recognition
- translation
- voice output
- history
- hardware communication

---

## 35.17 Progressive Web App

The React frontend could become a PWA with:

- installable app behavior
- cached assets
- improved loading
- app-like desktop/mobile use

---

## 35.18 Production Security

A larger public deployment should add:

- authentication where appropriate
- API rate limiting
- request validation
- strict CORS
- usage monitoring
- abuse protection
- structured logging
- alerts

This is especially important because AI API calls consume quota and potentially paid resources.

---

## 35.19 Automated Testing and CI/CD

Future engineering work could add:

- backend unit tests
- API integration tests
- frontend component tests
- browser end-to-end tests
- ML regression tests
- GitHub Actions
- automated deployment checks
- code-quality checks

---

## 35.20 Docker / Containerization

The **current project does not use Docker**.

Docker could be introduced later to package the FastAPI backend and Python dependencies into a reproducible container.

```text
Source Code
→ Docker Build
→ Container Image
→ Cloud Deployment
```

Benefits:

- consistent Python runtime
- reproducible dependencies
- easier migration between providers
- clearer separation between app and host environment

Docker is therefore a future improvement, not a technology used in the current implementation.

---

## 35.21 Accessibility-Focused User Research

A long-term version should be evaluated with actual Arabic Sign Language users and specialists.

Their feedback would be important for:

- sign accuracy
- linguistic correctness
- usability
- cultural appropriateness
- accessibility
- real communication needs

---

## Long-Term Vision

The long-term goal is to evolve the prototype into a multilingual intelligent communication platform.

```text
Real-Time Arabic Sign Language
→ Computer Vision
→ AI Recognition
→ Arabic Language Understanding
→ Multilingual Translation
→ Speech / Text / Physical Display
```

Potential application environments:

- airports
- hotels
- hospitals
- tourism
- government service centers
- education
- customer service
- international travel

---

# 36. Reproducibility Note

The current repository/runtime artifacts do not document the original exact:

```text
dataset size
samples per class
train/test split
random seed
RandomForest hyperparameters
first training-script filename
```

Those values should not be invented in technical documentation.

For complete reproducibility, a future repository revision should include:

```text
ml/data/
ml/train_static_classifier.py
ml/evaluate_static_classifier.py
```

with documented:

- dataset statistics
- class counts
- train/test split
- random seed
- hyperparameters
- evaluation metrics
- confusion matrix

---

# 37. Engineering Summary

This project demonstrates a complete AI engineering lifecycle:

```text
Gesture / Data Collection
        ↓
Hand Landmark Extraction
        ↓
Feature Engineering
        ↓
Machine Learning
        ↓
Real-Time Computer Vision
        ↓
FastAPI REST Backend
        ↓
Gemini Translation
        ↓
React Web Interface
        ↓
Speech / Web Serial
        ↓
Arduino + Physical LCD
        ↓
Cloud Deployment
```

The project demonstrates practical work across:

- AI / Machine Learning Engineering
- Computer Vision
- Feature Engineering
- Full-Stack Web Development
- REST API Design
- Generative AI Integration
- Frontend State Management
- Browser Hardware APIs
- Embedded Systems
- Arduino
- Cloud Deployment
- Version Control
- Debugging across software and hardware boundaries

---

## Author

Wejdan AlHarby

---

This project was developed as part of my Robotics Engineering with Intelligent Methods internship, where I applied Computer Vision and Machine Learning techniques and implemented hardware integration using USB serial communication, connecting a web application to an Arduino Uno and an I2C LCD display.
