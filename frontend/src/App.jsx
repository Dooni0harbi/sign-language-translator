import { useEffect, useRef, useState } from "react";
import {
  Camera,
  Languages,
  Plane,
  Volume2,
  Monitor,
  Sparkles,
  Hand,
} from "lucide-react";
import { motion } from "framer-motion";

const API_BASE_URL = "https://sign-language-translator-backend-dxbg.onrender.com";

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const timerRef = useRef(null);
  const sendingFrameRef = useRef(false);

  const [isRunning, setIsRunning] = useState(false);
  const [selectedLanguage, setSelectedLanguage] =
    useState("english");

  const [prediction, setPrediction] = useState("");
  const [confidence, setConfidence] = useState(0);
  const [arabicText, setArabicText] = useState("");
  const [translatedText, setTranslatedText] =
    useState("");

  const [progress, setProgress] = useState(0);
  const [requiredFrames, setRequiredFrames] =
    useState(1);

  const [statusMessage, setStatusMessage] =
    useState("جاهز لبدء الترجمة");

  useEffect(() => {
    return () => {
      stopFrameTimer();
      stopCameraStream();
    };
  }, []);

  async function toggleTranslation() {
    if (isRunning) {
      await stopTranslation();
    } else {
      await startTranslation();
    }
  }

  async function startTranslation() {
    resetResults();
    setStatusMessage("جاري تشغيل الكاميرا...");

    try {
      const stream =
        await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: "user",
          },
          audio: false,
        });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      const response = await fetch(`${API_BASE_URL}/api/start`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          mode: selectedLanguage,
        }),
      });

      const result = await response.json();

      if (!response.ok || !result.success) {
        throw new Error(
          result.error ||
            "تعذر بدء جلسة الترجمة"
        );
      }

      setIsRunning(true);
      setStatusMessage(
        "اعرضي الإشارة داخل إطار الكاميرا"
      );

      timerRef.current = window.setInterval(
        sendCameraFrame,
        40
      );
    } catch (error) {
      console.error(error);

      stopCameraStream();

      setStatusMessage(
        error.message || "تعذر تشغيل الكاميرا"
      );
    }
  }

  async function sendCameraFrame() {
    if (
      sendingFrameRef.current ||
      !videoRef.current ||
      videoRef.current.readyState < 2
    ) {
      return;
    }

    sendingFrameRef.current = true;

    try {
      const canvas = canvasRef.current;
      const video = videoRef.current;

      if (!canvas || !video) {
        return;
      }

      const width = 256;
      const height = 192;

      canvas.width = width;
      canvas.height = height;

      const context = canvas.getContext("2d");

      if (!context) {
        return;
      }

      context.save();
      context.translate(width, 0);
      context.scale(-1, 1);

      context.drawImage(
        video,
        0,
        0,
        width,
        height
      );

      context.restore();

      const image = canvas.toDataURL(
        "image/jpeg",
        0.55
      );

      const response = await fetch(`${API_BASE_URL}/api/frame`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ image }),
      });

      const result = await response.json();

      if (!response.ok || !result.success) {
        throw new Error(
          result.error ||
            "تعذر معالجة صورة الكاميرا"
        );
      }

      updateRecognitionResult(result);
    } catch (error) {
      console.error(error);
      setStatusMessage(
        "تعذر تحليل أحد إطارات الكاميرا"
      );
    } finally {
      sendingFrameRef.current = false;
    }
  }

  function updateRecognitionResult(result) {
    setPrediction(
      formatPrediction(result.prediction)
    );

    setConfidence(result.confidence || 0);
    setArabicText(result.arabic_preview || "");

    setProgress(result.progress || 0);

    setRequiredFrames(
      Math.max(result.required_frames || 1, 1)
    );

    if (result.progress_type === "motion") {
      setStatusMessage(
        "جاري التحقق من كلمة اسمي..."
      );
    } else if (result.current_candidate) {
      setStatusMessage(
        `جاري التحقق من ${
          formatPrediction(
            result.current_candidate
          ) || "الإشارة"
        }`
      );
    } else if (result.status_message) {
      setStatusMessage(result.status_message);
    } else {
      setStatusMessage(
        "اعرضي الإشارة داخل إطار الكاميرا"
      );
    }
  }

  async function stopTranslation() {
    stopFrameTimer();

    setStatusMessage("جاري ترجمة النص باستخدام الذكاء الاصطناعي...");

    try {
      const response = await fetch(`${API_BASE_URL}/api/stop`, {
        method: "POST",
      });

      const result = await response.json();

      if (!response.ok || !result.success) {
        throw new Error(
          result.error ||
            "تعذر إنهاء الترجمة"
        );
      }

      setArabicText(result.arabic || "");

      if (selectedLanguage === "english") {
        setTranslatedText(result.english || "");

        if (result.english) {
          setStatusMessage(
            "تمت الترجمة وتشغيل الصوت"
          );
        } else {
          setStatusMessage(
            result.message ||
              "لم تتوفر ترجمة إنجليزية"
          );
        }
      }

      if (selectedLanguage === "german") {
        setTranslatedText(result.german || "");

        if (result.german) {
          setStatusMessage(
            "تمت الترجمة الألمانية"
          );
        } else {
          setStatusMessage(
            result.message ||
              "لم تتوفر ترجمة ألمانية"
          );
        }
      }
    } catch (error) {
      console.error(error);

      setStatusMessage(
        error.message || "فشلت عملية الترجمة"
      );
    } finally {
      setIsRunning(false);
      stopCameraStream();
    }
  }

  function speakText(text) {
    if (
      !text ||
      !("speechSynthesis" in window)
    ) {
      return;
    }

    window.speechSynthesis.cancel();

    const speech =
      new SpeechSynthesisUtterance(text);

    speech.lang = "en-US";
    speech.rate = 0.9;
    speech.pitch = 1;

    window.speechSynthesis.speak(speech);
  }

  function stopFrameTimer() {
    if (timerRef.current) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }

  function stopCameraStream() {
    if (streamRef.current) {
      streamRef.current
        .getTracks()
        .forEach((track) => track.stop());
    }

    streamRef.current = null;

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }

  function resetResults() {
    setPrediction("");
    setConfidence(0);
    setArabicText("");
    setTranslatedText("");
    setProgress(0);
    setRequiredFrames(1);
  }

  function selectLanguage(language) {
    if (isRunning) {
      return;
    }

    setSelectedLanguage(language);
    setTranslatedText("");
  }

  const progressPercentage = Math.min(
    (progress / requiredFrames) * 100,
    100
  );

  return (
    <main className="app-shell">
      <FloatingLetters />

      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">
            <Hand size={26} />
          </div>

          <div>
            <h1>مترجم لغة الإشارة العربية</h1>
            <p>
              Arabic Sign Language Translator
            </p>
          </div>
        </div>

        <div className="travel-badge">
          <Plane size={20} />
          <span>تواصل بلا حدود</span>
        </div>
      </header>

      <section className="hero">
        <motion.div
          className="hero-copy"
          initial={{ opacity: 0, y: 25 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
        >
          <span className="eyebrow">
            <Sparkles size={17} />
            ترجمة ذكية أثناء السفر
          </span>

          <h2>
            إشارتك العربية
            <br />
            تصل إلى العالم
          </h2>

          <p>
            حوّلي لغة الإشارة العربية إلى نص
            مترجم وصوت مسموع، أو أرسلي الترجمة
            إلى شاشة خارجية.
          </p>

          <div className="language-buttons">
            <button
              disabled={isRunning}
              className={
                selectedLanguage === "english"
                  ? "language-button active english"
                  : "language-button english"
              }
              onClick={() =>
                selectLanguage("english")
              }
            >
              <Volume2 size={19} />
              English Voice
            </button>

            <button
              disabled={isRunning}
              className={
                selectedLanguage === "german"
                  ? "language-button active german"
                  : "language-button german"
              }
              onClick={() =>
                selectLanguage("german")
              }
            >
              <Monitor size={19} />
              German LCD
            </button>
          </div>
        </motion.div>

        <motion.div
          className="travel-visual"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{
            duration: 0.7,
            delay: 0.15,
          }}
        >
          <div className="globe">
            <div className="globe-line line-one" />
            <div className="globe-line line-two" />
            <div className="globe-line line-three" />

            <Plane
              className="flying-plane"
              size={46}
            />

            <span className="globe-letter letter-alef">
              أ
            </span>

            <span className="globe-letter letter-waw">
              و
            </span>

            <span className="globe-letter letter-jeem">
              ج
            </span>
          </div>
        </motion.div>
      </section>

      <section className="workspace">
        <div className="camera-card">
          <div className="card-heading">
            <div>
              <span className="section-label">
                LIVE CAMERA
              </span>

              <h3>
                اعرضي الإشارة داخل الإطار
              </h3>
            </div>

            <div
              className={
                isRunning
                  ? "status-pill online"
                  : "status-pill"
              }
            >
              <span />
              {isRunning ? "مباشر" : "جاهز"}
            </div>
          </div>

          <div className="camera-frame">
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
            />

            <canvas
              ref={canvasRef}
              style={{ display: "none" }}
            />

            {!isRunning && (
              <div className="camera-placeholder">
                <div className="placeholder-icon">
                  <Camera size={42} />
                </div>

                <h4>الكاميرا متوقفة</h4>

                <p>
                  اختاري اللغة ثم اضغطي على
                  زر البدء
                </p>
              </div>
            )}

            <div className="scan-corner top-left" />
            <div className="scan-corner top-right" />
            <div className="scan-corner bottom-left" />
            <div className="scan-corner bottom-right" />
          </div>

          <button
            className={
              isRunning
                ? "camera-button stop"
                : "camera-button"
            }
            onClick={toggleTranslation}
          >
            <Camera size={21} />

            {isRunning
              ? "إيقاف وترجمة"
              : "بدء الترجمة"}
          </button>

          <p
            style={{
              textAlign: "center",
              marginBottom: 0,
            }}
          >
            {statusMessage}
          </p>
        </div>

        <div className="results-column">
          <div className="prediction-card">
            <div className="prediction-icon">
              <Languages size={25} />
            </div>

            <div>
              <span>الإشارة الحالية</span>

              <strong>
                {prediction ||
                  "بانتظار الإشارة"}
              </strong>
            </div>

            <div className="confidence-ring">
              {confidence}%
            </div>
          </div>

          <div className="progress-card">
            <div className="progress-header">
              <span>
                جاري التحقق من الإشارة
              </span>

              <span>
                {Math.round(progressPercentage)}%
              </span>
            </div>

            <div className="progress-track">
              <motion.div
                className="progress-value"
                animate={{
                  width: `${progressPercentage}%`,
                }}
                transition={{ duration: 0.2 }}
              />
            </div>
          </div>

          <div className="translation-card arabic">
            <span className="translation-label">
              النص العربي
            </span>

            <p>
              {arabicText ||
                "سيظهر النص بعد التعرّف"}
            </p>
          </div>

          <div
            className={
              selectedLanguage === "english"
                ? "translation-card translated english-card"
                : "translation-card translated german-card"
            }
          >
            <span className="translation-label">
              {selectedLanguage === "english"
                ? "English Translation"
                : "German Translation"}
            </span>

            <p>
              {translatedText ||
                "ستظهر الترجمة بعد اكتمال التعرّف"}
            </p>

            <button
              className="result-action"
              disabled={!translatedText}
              onClick={() => {
                if (
                  selectedLanguage === "english"
                ) {
                  speakText(translatedText);
                }
              }}
            >
              {selectedLanguage ===
              "english" ? (
                <>
                  <Volume2 size={18} />
                  تشغيل الصوت بالإنجليزية
                </>
              ) : (
                <>
                  <Monitor size={18} />
                  إرسال إلى الشاشة
                </>
              )}
            </button>
          </div>
        </div>
      </section>
    </main>
  );
}

function formatPrediction(label) {
  const labels = {
    ana: "أنا",
    esme_motion: "اسمي",
    waw: "و",
    jeem: "ج",
    dal: "د",
    alif: "ا",
    noon: "ن",
    checking_esme: "جاري فحص اسمي",
    other_motion: "حركة غير معروفة",
  };

  return labels[label] || label || "";
}

function FloatingLetters() {
  const letters = [
    { value: "أ", className: "float-one" },
    { value: "ب", className: "float-two" },
    { value: "ج", className: "float-three" },
    { value: "د", className: "float-four" },
    { value: "و", className: "float-five" },
    { value: "ن", className: "float-six" },
  ];

  return (
    <div
      className="floating-letters"
      aria-hidden="true"
    >
      {letters.map((letter) => (
        <span
          key={letter.className}
          className={letter.className}
        >
          {letter.value}
        </span>
      ))}
    </div>
  );
}

export default App;