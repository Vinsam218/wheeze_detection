# Wheeze Detection from Respiratory Sounds

**Project type:** Research prototype for binary wheeze detection from respiratory audio.

**Model plan**
- **Option A — baseline:** Hand-crafted audio features + a traditional classifier.
- **Option B — primary candidate:** Compact CNN trained on log-Mel spectrograms.
- **Augmentation:** Compare original training data, real-audio augmentation, and synthetic overlays in separate experiments.
- **Evaluation:** Patient-level splits created before augmentation to prevent leakage.

> **Important:** This is a research prototype, not a medical device or diagnostic tool. Detecting wheeze is not the same as diagnosing asthma or estimating attack severity.

## 1. Problem statement

Wheezing is a respiratory sound that can occur in asthma and other respiratory conditions. Automated detection is challenging because respiratory audio varies across people, recording devices, environments, and respiratory cycles.

This project investigates whether machine-learning models can distinguish respiratory cycles annotated as wheeze-positive from cycles without an annotated wheeze. Using the ICBHI 2017 Respiratory Sound Database, it compares a feature-based baseline with a compact CNN trained on time-frequency representations. It will also investigate whether audio augmentation improves robustness and whether the trained model can be demonstrated through a browser-based app using a phone or laptop microphone.

### Research question

**How accurately can audio-based machine-learning models detect wheeze in annotated respiratory-sound recordings, and how do feature-based and CNN approaches compare when evaluated on patients excluded from training?**

### Supporting questions

1. How well does a traditional classifier perform with hand-crafted audio features?
2. Does a CNN trained on log-Mel spectrograms improve performance?
3. Does real-audio augmentation improve generalisation?
4. Do synthetic overlays add benefit beyond real-audio augmentation?
5. How does performance change on ordinary phone/laptop microphone recordings?

## 2. Aim and objectives

### Aim

Develop and evaluate a reproducible respiratory-audio pipeline for wheeze detection, then demonstrate inference through a web application.

### Objectives

- Audit the dataset, annotations, class balance, and audio quality.
- Build a cycle-level manifest linking audio cycles to labels, recordings, and patients.
- Create patient-level train/validation/test splits before augmentation.
- Train a feature-based baseline and a compact CNN.
- Compare original-only training with real-audio augmentation and synthetic overlays.
- Evaluate using suitable metrics and prevent test-set leakage.
- Package inference behind an API and create a microphone/audio-upload web interface.
- Document limitations and the need for independent validation.

## 3. Dataset

### ICBHI 2017 Respiratory Sound Database

The project currently uses a Kaggle-hosted copy of the ICBHI 2017 Respiratory Sound Database. The local development path is:

```text
/home/vinsam/.cache/kagglehub/datasets/nimalanparameshwaran/icbhi-2017-challenge-respiratory-sound-database/versions/1/ICBHI_final_database
```

The folder contains WAV recordings and annotation text files. Annotation rows provide respiratory-cycle start/end times and crackle/wheeze indicators. Cite the original ICBHI resource and check the applicable dataset terms; the Kaggle copy is only a distribution route.

### Preliminary audit

Current notebook exploration reported:

- **6,898** annotated respiratory cycles
- **126** patients
- **1,392** wheeze-positive cycles (about **20.2%**)
- **5,506** cycles without an annotated wheeze (about **79.8%**)

These are preliminary counts. Recompute them from the final manifest before publishing results.

### Target label

The first task is **binary detection at respiratory-cycle level**:

| Label | Definition |
|---|---|
| `1` | Cycle annotation indicates wheeze |
| `0` | Cycle annotation does not indicate wheeze |

Retain crackle annotations for auditing, but do not use crackle as the target in this first task. “No annotated wheeze” does not necessarily mean a healthy recording or the absence of other abnormal sounds.

**Scope boundary:** This first model does not classify wheeze intensity, diagnose asthma, detect an asthma attack, or estimate attack severity. Those require separate labels, clinically meaningful definitions, expert input, and independent validation.

## 4. Methodology

1. Audit audio/annotation pairing, sample rates, durations, annotation boundaries, and unreadable files.
2. Build a master manifest with recording ID, patient ID, cycle start/end, duration, wheeze label, crackle label, and source path.
3. Inspect class balance and representative examples.
4. Assign patients to train, validation, and test sets. Keep all cycles and derivatives from a patient in one partition.
5. Apply consistent preprocessing for both training and inference.
6. Train baseline A0 and primary model B0.
7. Apply augmentation to training data only.
8. Compare models using the same patient splits and evaluation procedure.
9. Package preprocessing and inference for the API.
10. Evaluate microphone/domain-shift limitations before making real-world robustness claims.

## 5. Models

### Option A — feature-based baseline

This is the reference model: simpler, easier to inspect, and useful for establishing a first benchmark.

Candidate features:
- MFCCs, summarised across each cycle
- Spectral centroid and bandwidth
- Zero-crossing rate
- RMS energy

Candidate classifiers:
- Logistic regression
- Random forest

Start with one documented feature configuration and one classifier. Fit any learned scaling/preprocessing on training data only.

### Option B — primary candidate: compact CNN

Convert each cycle into a **log-Mel spectrogram** and train a small CNN to predict the binary label.

Pipeline:
1. Decode and preprocess the cycle.
2. Generate a log-Mel spectrogram using fixed, documented parameters.
3. Apply a consistent normalisation strategy.
4. Run the spectrogram through the CNN.
5. Produce a wheeze score/probability.
6. Select the decision threshold on validation data, then freeze it before test evaluation.

Record sample rate, input dimensions, spectrogram parameters, architecture, random seeds, and training settings.

### Planned experiments

| ID | Model | Training data | Purpose |
|---|---|---|---|
| A0 | Feature baseline | Original only | Establish baseline |
| A1 | Feature baseline | Real-audio augmentation | Measure augmentation effect |
| A2 | Feature baseline | Real augmentation + synthetic overlays | Test added value of overlays |
| B0 | CNN | Original only | Establish CNN reference |
| B1 | CNN | Real-audio augmentation | Measure CNN augmentation effect |
| B2 | CNN | Real augmentation + synthetic overlays | Test added value of overlays |

Complete A0 and B0 first. Continue to A1/B1 and then A2/B2 if time and initial results justify it. Use the same patient splits across experiments.

## 6. Augmentation and synthetic data

Augmentation is a robustness experiment, not a way to manufacture clinical ground truth.

### Real-audio augmentation

Potential transformations include low-level additive noise, small gain changes, and carefully controlled shifts/crops when the label remains valid. Keep source files unchanged. Save augmented audio separately and record the source cycle, transformation, parameters, and random seed.

### Synthetic overlays

Synthetic overlays may combine respiratory cycles with documented background noise or other audio components. Keep these separate from real patient data. Do not add a synthetic wheeze to a negative cycle and treat the result as ground truth unless the experiment explicitly defines and validates that labelling process.

Never augment validation/test data. No augmented derivative of a source cycle or patient may cross the split boundary.

## 7. Evaluation and leakage prevention

### Patient-level splits

Cycles from the same patient can share acoustic and recording characteristics. Randomly splitting cycles can inflate scores. Split by **patient before augmentation**, save the patient IDs in versioned split files, and keep the test set untouched until final evaluation.

### Metrics

Report more than accuracy:
- Sensitivity/recall
- Specificity
- Precision
- F1 score
- Confusion matrix
- ROC-AUC and, where appropriate, PR-AUC
- Model size and inference latency

Document the decision threshold and how it was selected. Report per-class results and patient-level variation where feasible. A high score on one dataset does not establish clinical validity.

## 8. Repository structure

```text
wheeze-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
├── docs/
│   ├── PROJECT_PIPELINE.md
│   ├── DATASET_CARD.md
│   ├── METHODS.md
│   ├── EXPERIMENTS.md
│   ├── MODEL_CARD.md
│   └── DEPLOYMENT.md
├── notebooks/
│   └── crobemlmine.ipynb
├── src/
│   ├── data/
│   ├── augmentation/
│   ├── features/
│   ├── models/
│   └── api/
├── web/
├── tests/
├── experiments/
└── artifacts/
```

Keep reusable pipeline code in `src/`; use notebooks for exploration and plots.

### Local data layout

Keep datasets outside the public source repository:

```text
~/datasets/wheeze-project/
├── raw/ICBHI_final_database/
├── processed/manifests/
├── processed/cycles/
├── processed/patient_splits/
├── augmented/real_audio/
├── augmented/synthetic/
└── evaluation/
    ├── predictions/
    └── reports/
```

Audio can be large and may have redistribution restrictions. Do not commit raw audio, generated WAVs, private recordings, credentials, or sensitive metadata to public GitHub by default. Check dataset terms before redistributing any source or derived data.

### Suggested manifest fields

`recording_id`, `patient_id`, `audio_path`, `start_seconds`, `end_seconds`, `duration_seconds`, `wheeze_label`, `crackle_label`, `split`, `source_cycle_id`, `augmentation_type`, `augmentation_parameters`, `random_seed`, and `dataset_version` or source hash.

Keep original rows distinguishable from augmented rows and save a separate split manifest.

## 9. Web app and deployment

The planned demo lets a user record from a phone/laptop microphone or upload audio, then request an experimental prediction.

### Proposed architecture

- **Frontend:** React + TypeScript; record, stop, playback, upload, and result states.
- **Browser audio:** `getUserMedia` and `MediaRecorder`, with permission/error handling. Microphone access generally requires HTTPS or a secure local context.
- **Backend:** Python API, such as FastAPI.
- **Inference:** Same audio decoding, preprocessing, and model configuration as training.
- **Output:** `Wheeze pattern detected`, `No wheeze detected`, or `Unable to assess`, with a clear experimental-use notice.

### Deployment cautions

- Phone/laptop microphones differ from clinical stethoscopes and the ICBHI recording setup; domain shift is a major risk.
- Test on representative target-device recordings before claiming reliability outside the dataset.
- Do not use the app to diagnose asthma, estimate attack severity, advise medication changes, or replace medical care.
- Do not store microphone recordings by default. If recordings are collected for research, obtain appropriate consent, minimise collection, secure access, and document retention/deletion.
- Handle silence, clipping, unsupported formats, short audio, noise, and server errors.
- Measure end-to-end inference latency.

## 10. Reproducibility

For each experiment, record the experiment ID/date, dataset version and manifest hash, patient split version, model/configuration, augmentation parameters, random seeds, software versions, threshold-selection method, metrics, confusion matrix, predictions, and artifact checksum.

Keep experiment outputs separate; do not overwrite previous results. Compare models on the same test cases.

## 11. Roadmap

1. Create repository structure, README, `.gitignore`, and environment instructions.
2. Audit audio/annotation pairs and regenerate dataset counts.
3. Build the cycle manifest and patient-level splits.
4. Implement and evaluate **A0**, the feature baseline.
5. Implement and evaluate **B0**, the CNN.
6. Run real-audio augmentation experiments; test synthetic overlays only if justified.
7. Compare models on fixed patient splits.
8. Refactor preprocessing/inference into reusable code.
9. Build the API and browser frontend.
10. Evaluate consented target-device recordings and document limitations.

## 12. Limitations and responsible use

- Labels represent annotated wheeze at cycle level, not asthma-attack severity.
- Annotations and recording conditions may be imperfect.
- Evaluation on one dataset does not prove generalisation to other populations, settings, or consumer microphones.
- Class imbalance can make accuracy misleading.
- Synthetic audio is not a substitute for diverse real-world data.
- Research performance does not establish safety, clinical validity, or regulatory compliance.

Any future wheeze-intensity or asthma-severity task must be treated as a separate project with suitable labels, clinically grounded definitions, expert involvement, and independent validation.

## 13. References

- ICBHI 2017 Challenge: https://bhichallenge.med.auth.gr/welcome
- ICBHI challenge/database resources: https://bhichallenge.med.auth.gr/node
- Scikit-learn cross-validation guide: https://scikit-learn.org/stable/modules/cross_validation.html
- MDN — MediaRecorder: https://developer.mozilla.org/en-US/docs/Web/API/MediaRecorder
- MDN — MediaStream Recording API: https://developer.mozilla.org/en-US/docs/Web/API/MediaStream_Recording_API/Using_the_MediaStream_Recording_API

Check original dataset terms and applicable ethics, privacy, and deployment requirements before redistributing data or collecting new recordings.

## Current status

**Stage:** Dataset exploration and pipeline planning.

**Next milestone:** Finalise the cycle-level manifest and patient-level splits, then implement A0 and B0 using the same split before prioritising augmentation experiments.
