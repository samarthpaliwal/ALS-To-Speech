# ALS-To-Speech

> The repository name is a typo for ASL (American Sign Language). The project has nothing to do with ALS.

> **Early project.** One of the first things I built, kept public as a record of where I started. Current work is on my [profile](https://github.com/samarthpaliwal).

<!-- TODO: add a screenshot as docs/screenshot.png and uncomment.
![Screenshot](docs/screenshot.png)
-->

A webcam-based American Sign Language (ASL) hand-sign recognizer written in Python. It uses MediaPipe Hands to extract 21 hand landmarks from each camera frame, stores those landmark vectors as training data, trains a scikit-learn Random Forest on them, and provides a classifier class that draws the predicted sign on a live frame. The repository name suggests a speech output stage, but no text-to-speech code is present; the project currently stops at recognizing the sign.

## Requirements

- Python 3.9 to 3.11 (MediaPipe wheels are not available for every newer version)
- A webcam
- Python packages: `opencv-python`, `mediapipe`, `numpy`, `scikit-learn`

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install opencv-python mediapipe numpy scikit-learn
python test_imports.py          # prints the Python path and OpenCV version to confirm the install
```

## Usage

The scripts are run individually from the repository root.

1. Check that the camera and hand tracking work:

   ```bash
   python hand_detection.py
   ```

   A window titled "ASL Hand Detection" opens showing the mirrored webcam feed with hand landmarks drawn on it. Press `q` to quit.

2. Collect training data:

   ```bash
   python capture_data.py
   ```

   The script walks through the letters A to Y (skipping J and Z, which require motion). For each letter it prints `Collecting for <letter>. Press 'c' to capture.`; hold the sign in front of the camera and press `c` once per sample until 100 samples are saved. Each sample is written as `data/<letter>/<n>.pkl` containing 63 floats (x, y, z for 21 landmarks).

3. Train the classifier:

   ```bash
   python train_model.py
   ```

   Loads everything under `data/`, does an 80/20 stratified split, trains a 100-tree Random Forest, prints `Accuracy: <score>`, and saves the model to `asl_rf_model.pkl`.

4. Live recognition: `asl_recognition.py` defines a `GestureClassifier` class but there is no script that opens the camera and calls it. To use it you would import it and loop over frames yourself:

   ```python
   import cv2
   from asl_recognition import GestureClassifier
   clf = GestureClassifier()
   cap = cv2.VideoCapture(0)
   while True:
       ok, frame = cap.read()
       label, frame = clf.predict(frame)
       cv2.imshow("ASL", frame)
       if cv2.waitKey(1) & 0xFF == ord("q"):
           break
   ```

   Note the mismatches described below before trying this.

## How it works

- `hand_detection.py` opens the default camera, runs MediaPipe Hands (one hand, 0.7 detection confidence) and draws the landmark skeleton on each frame.
- `capture_data.py` does the same detection but, on a `c` keypress, flattens the 21 landmarks into a 63-value vector and pickles it under a per-letter folder.
- `train_model.py` reads those pickles into feature/label arrays, trains `RandomForestClassifier(n_estimators=100)`, reports held-out accuracy, and pickles the model.
- `asl_recognition.py` loads a model from `./model.p` (expected to be a dict with a `"model"` key), normalizes landmark x/y coordinates relative to the hand's bounding box, pads the vector to 84 values, predicts a label, maps it through `labels_dict`, and draws a bounding box and label on the frame.

Known inconsistencies in the current code:

- `train_model.py` saves `asl_rf_model.pkl` (a bare classifier), but `asl_recognition.py` loads `model.p` and expects a dict containing a `"model"` key. The two files do not connect without a manual step.
- `capture_data.py` produces 63 features (x, y, z per landmark); `asl_recognition.py` builds 84 features (x, y per landmark, normalized, for up to two hands). A model trained with this repository's data will not accept the recognizer's input shape.
- `labels_dict.py` lists many word signs (dad, mom, water, thank you, numbers) that `capture_data.py` never collects; it appears to come from a different, larger dataset.
- No `data/`, `model.p` or `asl_rf_model.pkl` is committed, so training data must be collected locally first.

## Project layout

- `hand_detection.py`: webcam demo that draws MediaPipe hand landmarks.
- `capture_data.py`: interactive collector that saves 100 landmark samples per letter to `data/`.
- `train_model.py`: trains a Random Forest on `data/` and writes `asl_rf_model.pkl`.
- `asl_recognition.py`: `GestureClassifier` class that loads `model.p` and annotates frames with the predicted sign.
- `labels_dict.py`: identity mapping from model label to display string.
- `test_imports.py`: prints the Python executable and OpenCV version.
