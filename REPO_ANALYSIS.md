# Repository Analysis

## Overview
This repository is a prototype for a **multimodal music recommendation app** that combines:
- facial emotion prediction,
- age and gender prediction,
- indoor scene classification,
- YouTube search/playback for recommended songs.

The main runtime app is a Streamlit script (`emotion_age_recom_v1.py`).

## Project Structure (high level)
- `emotion_age_recom_v1.py`: main Streamlit application wiring all models and YouTube playback.
- `hello.py`: helper that queries YouTube Data API for songs based on emotion/age/gender/language.
- `realtime_age_classifier/real-time.py`: OpenCV webcam demo for age category prediction.
- `README.md`: project description, setup, and claimed results.
- `requirements.txt`: Python dependencies.
- `*.h5`, `*.pt`: pre-trained model artifacts.
- `*.ipynb`: notebooks for experiments/training.

## What works well
- End-to-end concept is clear: detect user context then recommend content.
- Includes model artifacts locally, so inference can run without retraining.
- Streamlit UI flow is straightforward (detect -> recommend -> play).
- YouTube duration filtering avoids extremely short/long videos.

## Key issues and risks

### 1) Hard-coded absolute Windows model paths
The app loads models from `D:/Music_v3/...`, which will fail on most environments.

**Impact:** app crashes at startup unless that exact local path exists.

### 2) Secret/API key committed in code
A YouTube API key appears directly in `emotion_age_recom_v1.py`.

**Impact:** credential leakage/security risk and potential quota abuse.

### 3) Logic bug: `recommended_genre` is ignored
In `hello.py`, `genres` is initially set from `recommended_genre` but immediately overwritten by emotion mapping.

**Impact:** indoor-scene recommendation has no effect.

### 4) Emotion class mismatch risk
`predict_emotion()` maps model outputs to 8 custom labels including `contempt`, while `hello.py` emotion mapping has 7 keys and no `contempt`.

**Impact:** potential fallback behavior and lower recommendation quality for unmatched labels.

### 5) Webcam capture assumptions
The app captures age/gender from webcam once at startup and assumes success.

**Impact:** fragile behavior on headless/cloud environments; `age`/`gender` may be undefined in failure paths.

### 6) No tests or validation harness
No automated tests for inference pipelines, preprocessing, or API integration.

**Impact:** regressions likely when changing models or preprocessing.

### 7) Large binaries in repo
Multiple `.h5`/`.pt` artifacts are tracked directly.

**Impact:** heavy clone size and difficult versioning unless managed with Git LFS/release assets.

## Priority fixes (recommended)
1. Replace absolute model paths with relative paths + existence checks.
2. Move API key to environment variables / Streamlit secrets.
3. Fix `hello.py` genre override bug so `recommended_genre` is honored.
4. Normalize emotion label set between model output and search mapping.
5. Add graceful fallbacks for missing webcam and file uploads.
6. Add minimal tests for preprocessing and recommendation logic.

## Suggested architecture improvements
- Split `emotion_age_recom_v1.py` into modules (`models.py`, `preprocess.py`, `recommend.py`, `ui.py`).
- Add config file for model paths/labels.
- Add typed dataclasses for prediction payloads.
- Cache loaded models (e.g., `@st.cache_resource`) to reduce startup cost.
- Add linting/formatting (`ruff`, `black`) and CI checks.

## Quick run notes
To run locally (after path and API-key cleanup):
```bash
pip install -r requirements.txt
streamlit run emotion_age_recom_v1.py
```

## Conclusion
The repository demonstrates a promising prototype, but it is not yet production-ready due to portability, security, and logic-consistency issues. With the fixes above, it can become a stable baseline for multimodal recommendation experimentation.
