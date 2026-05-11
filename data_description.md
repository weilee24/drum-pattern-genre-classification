# Data Description

## Dataset Source

This project uses the **Groove MIDI Dataset (GMD)** from Google Magenta. The dataset contains human-performed drum grooves with audio recordings, aligned MIDI files, and style labels.

The dataset is not included in this repository because of its size. The preprocessing notebook downloads it directly.

## Raw Dataset Summary

From the project preprocessing notebook:

- Original metadata rows: 1,150
- Rows with available audio files: 1,090
- Audio files used for feature extraction: 1,090
- Final saved feature table: 1,090 rows, 35 columns

## Target Labels

The original dataset includes the following style labels after extracting the main style name:

- funk
- soul
- hiphop
- pop
- rock
- jazz
- neworleans
- dance
- latin
- afrocuban
- reggae
- country
- gospel
- punk
- afrobeat
- blues
- middleeastern

For modeling, rare styles were merged into an `others` class. The final classification task used 14 classes.

Merged rare styles:

- dance
- afrobeat
- blues
- middleeastern

## Feature Extraction

Each audio file was loaded using `librosa` with:

- Sampling rate: 22,050 Hz
- Mono audio conversion

Extracted features:

| Feature Type | Features |
|---|---|
| Rhythm | tempo, onset rate |
| Time domain | zero-crossing rate, RMS energy |
| Spectral | spectral centroid, spectral bandwidth, spectral rolloff |
| Timbre | 13 MFCC means, 13 MFCC standard deviations |

## Data Splitting

The modeling notebook used a stratified split:

- Training: 70%
- Validation: 15%
- Test: 15%

The validation set contained 163 samples. The training set contained 763 samples.

## Scaling

All numeric features were standardized using `StandardScaler` fit only on the training data. The same scaler was then applied to validation and test data to avoid data leakage.

## Data Leakage Control

The project avoids data leakage by:

- Splitting data before fitting the scaler
- Fitting `StandardScaler` only on the training set
- Applying the trained scaler to validation and test sets

## Limitation

The dataset contains isolated drum performances rather than full songs. This makes it suitable for studying drum-pattern style recognition, but it does not directly represent full-track music genre classification.
