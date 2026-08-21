const cameraVideo =
  document.getElementById("cameraVideo");

const captureCanvas =
  document.getElementById("captureCanvas");

const cameraPlaceholder =
  document.getElementById("cameraPlaceholder");

const recordingBadge =
  document.getElementById("recordingBadge");

const englishButton =
  document.getElementById("englishButton");

const germanButton =
  document.getElementById("germanButton");

const statusText =
  document.getElementById("statusText");

const expectedText =
  document.getElementById("expectedText");

const predictionText =
  document.getElementById("predictionText");

const confidenceText =
  document.getElementById("confidenceText");

const progressBar =
  document.getElementById("progressBar");

const sequenceText =
  document.getElementById("sequenceText");

const arabicResult =
  document.getElementById("arabicResult");

const translatedResult =
  document.getElementById("translatedResult");

const translatedLanguageLabel =
  document.getElementById(
    "translatedLanguageLabel"
  );

const messageText =
  document.getElementById("messageText");


let cameraStream = null;
let frameTimer = null;
let activeMode = null;
let isRecording = false;
let sendingFrame = false;


async function toggleTranslation(mode) {
  if (!isRecording) {
    await startTranslation(mode);
    return;
  }

  if (activeMode === mode) {
    await stopTranslation();
  }
}


async function startTranslation(mode) {
  resetDisplayedResults();

  statusText.textContent =
    "Requesting camera access...";

  messageText.textContent = "";

  try {
    cameraStream =
      await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "user",
        },
        audio: false,
      });

    cameraVideo.srcObject = cameraStream;
    await cameraVideo.play();

    const response = await fetch(
      "/api/start",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ mode }),
      }
    );

    const result = await response.json();

    if (!response.ok || !result.success) {
      throw new Error(
        result.error ||
        "Could not start translation."
      );
    }

    activeMode = mode;
    isRecording = true;

    updateRecordingInterface(mode);

    // نحاول الإرسال كل 50ms.
    // لو السيرفر ما زال يعالج الفريم السابق،
    // لن نرسل فريمًا جديدًا حتى ينتهي.
    frameTimer = window.setInterval(
      sendCameraFrame,
      50
    );

  } catch (error) {
    console.error(error);

    stopCameraStream();

    statusText.textContent =
      "Could not start the camera.";

    messageText.textContent =
      error.message;
  }
}


async function sendCameraFrame() {
  if (
    !isRecording ||
    sendingFrame ||
    cameraVideo.readyState < 2
  ) {
    return;
  }

  sendingFrame = true;

  try {
    // نقلل حجم الصورة لتسريع MediaPipe والإرسال
    const width = 320;
    const height = 240;

    captureCanvas.width = width;
    captureCanvas.height = height;

    const context =
      captureCanvas.getContext("2d");

    context.save();
    context.translate(width, 0);
    context.scale(-1, 1);

    context.drawImage(
      cameraVideo,
      0,
      0,
      width,
      height
    );

    context.restore();

    const image = captureCanvas.toDataURL(
      "image/jpeg",
      0.65
    );

    const response = await fetch(
      "/api/frame",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ image }),
      }
    );

    const result = await response.json();

    if (!result.success) {
      throw new Error(
        result.error ||
        "Frame processing failed."
      );
    }

    updateRecognitionStatus(result);

  } catch (error) {
    console.error(error);

    messageText.textContent =
      "A camera frame could not be processed.";

  } finally {
    sendingFrame = false;
  }
}


function updateRecognitionStatus(result) {
  statusText.textContent =
    "Recognizing any trained sign...";

  expectedText.textContent =
    "ANY TRAINED SIGN";

  predictionText.textContent =
    formatLabel(result.prediction);

  confidenceText.textContent =
    `${result.confidence || 0}%`;

  sequenceText.textContent =
    result.sequence_text ||
    "No signs recognized yet";

  arabicResult.textContent =
    result.arabic_preview || "-";

  const requiredFrames =
    Math.max(result.required_frames || 1, 1);

  const currentProgress =
    Math.max(result.progress || 0, 0);

  const percentage = Math.min(
    (currentProgress / requiredFrames) * 100,
    100
  );

  progressBar.style.width =
    `${percentage}%`;

  if (result.progress_type === "motion") {
    messageText.textContent =
      "Checking ESME...";
  } else if (result.current_candidate) {
    messageText.textContent =
      `Hold ${formatLabel(
        result.current_candidate
      )} until the bar is full.`;
  } else {
    messageText.textContent =
      result.status_message ||
      "Show any trained sign.";
  }
}


async function stopTranslation() {
  clearInterval(frameTimer);
  frameTimer = null;

  statusText.textContent =
    "Finishing translation...";

  try {
    const response = await fetch(
      "/api/stop",
      {
        method: "POST",
      }
    );

    const result = await response.json();

    isRecording = false;
    stopCameraStream();
    restoreButtons();

    if (!result.completed) {
      statusText.textContent =
        "No signs were recorded.";

      messageText.textContent =
        result.message ||
        "Perform at least one trained sign.";

      activeMode = null;
      return;
    }

    sequenceText.textContent =
      result.sequence_text ||
      "No signs recognized";

    arabicResult.textContent =
      result.arabic || "-";

    if (result.mode === "english") {
      translatedLanguageLabel.textContent =
        "English";

      if (result.english) {
        translatedResult.textContent =
          result.english;

        statusText.textContent =
          "English translation completed.";

        messageText.textContent = "";

        speakEnglish(result.english);
      } else {
        translatedResult.textContent =
          "Translation not configured";

        statusText.textContent =
          "Signs were recorded.";

        messageText.textContent =
          result.message ||
          "This sentence does not have an English translation yet.";
      }
    }

    if (result.mode === "german") {
      translatedLanguageLabel.textContent =
        "German";

      if (result.german) {
        translatedResult.textContent =
          result.german;

        statusText.textContent =
          "German translation completed.";

        messageText.textContent =
          "Arduino LCD connection will be added next.";
      } else {
        translatedResult.textContent =
          "Translation not configured";

        statusText.textContent =
          "Signs were recorded.";

        messageText.textContent =
          result.message ||
          "This sentence does not have a German translation yet.";
      }
    }

    activeMode = null;

  } catch (error) {
    console.error(error);

    isRecording = false;
    stopCameraStream();
    restoreButtons();

    statusText.textContent =
      "Translation could not be completed.";

    messageText.textContent =
      error.message;
  }
}


function speakEnglish(text) {
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
  speech.volume = 1;

  window.speechSynthesis.speak(speech);
}


function updateRecordingInterface(mode) {
  cameraVideo.classList.add("active");
  cameraPlaceholder.classList.add("hidden");
  recordingBadge.classList.remove("hidden");

  statusText.textContent =
    "Recognizing any trained sign...";

  expectedText.textContent =
    "ANY TRAINED SIGN";

  if (mode === "english") {
    englishButton.textContent =
      "Stop & Translate to English";

    englishButton.classList.add(
      "recording"
    );

    germanButton.disabled = true;
  }

  if (mode === "german") {
    germanButton.textContent =
      "Stop & Translate to German";

    germanButton.classList.add(
      "recording"
    );

    englishButton.disabled = true;
  }
}


function restoreButtons() {
  englishButton.textContent =
    "Start English Translation";

  germanButton.textContent =
    "Start German Translation";

  englishButton.classList.remove(
    "recording"
  );

  germanButton.classList.remove(
    "recording"
  );

  englishButton.disabled = false;
  germanButton.disabled = false;

  recordingBadge.classList.add("hidden");
  cameraVideo.classList.remove("active");
  cameraPlaceholder.classList.remove("hidden");
}


function stopCameraStream() {
  if (cameraStream) {
    for (const track of cameraStream.getTracks()) {
      track.stop();
    }
  }

  cameraStream = null;
  cameraVideo.srcObject = null;
}


function resetDisplayedResults() {
  expectedText.textContent = "-";
  predictionText.textContent = "-";
  confidenceText.textContent = "0%";

  progressBar.style.width = "0%";

  sequenceText.textContent =
    "No signs recognized yet";

  arabicResult.textContent = "-";
  translatedResult.textContent = "-";

  translatedLanguageLabel.textContent =
    "Translation";

  messageText.textContent = "";
}


function formatLabel(label) {
  const labels = {
    checking_esme: "CHECKING ESME...",
    esme_motion: "ESME",
    other_motion: "NOT ESME",
    ana: "ANA",
    waw: "WAW",
    jeem: "JEEM",
    dal: "DAL",
    alif: "ALIF",
    noon: "NOON",
  };

  if (!label) {
    return "-";
  }

  return labels[label] ||
    String(label)
      .replaceAll("_", " ")
      .toUpperCase();
}


englishButton.addEventListener(
  "click",
  () => toggleTranslation("english")
);

germanButton.addEventListener(
  "click",
  () => toggleTranslation("german")
);
