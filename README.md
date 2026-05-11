# Bayesian Modeling of Drum Pattern Styles

This project is a **14-class drum-style classification task** using **drum-only audio**. The model does not use melody, vocals, lyrics, harmony, artist metadata, or full-song production context. It only uses rhythmic and spectral features extracted from isolated drum performances.

Because the task has **14 possible classes**, a strict top-1 accuracy of around **35% to 40%** should not be interpreted like a binary classification score. A balanced random guess would be about **7.1%**, and many classes are rhythmically similar or underrepresented. The more informative result is that the Bayesian models clearly outperform the simple Naive Bayes baseline and reach up to about **55% top-2 accuracy**, meaning the correct style is often among the model's two most probable predictions.

The goal of this project is not to claim perfect genre recognition. The goal is to test how much stylistic information can be recovered from drum patterns alone, while using Bayesian models to expose uncertainty and model limitations.

## Project Motivation

Music genre classification is difficult because modern genres often overlap. Many styles share similar production techniques, melodic patterns, and instrumentation. Drum patterns, however, often preserve rhythmic and stylistic cues.

The main question of this project is:

> Can musical style be classified using only drum-pattern features extracted from audio?

This setup is intentionally restrictive. Classifying 14 styles from drum-only audio is harder than classifying full songs, but it isolates whether rhythmic and timbral drum features carry enough signal for style prediction.

## Dataset

- Source: Groove MIDI Dataset (GMD), Google Magenta
- Original metadata: 1,150 entries
- Usable audio recordings: 1,090 files
- Final feature table: 1,090 rows and 35 columns
- Final task: **14-class classification** after merging rare styles into an `others` class
- Split: 70% training, 15% validation, 15% test, using stratified sampling

Rare classes such as `dance`, `afrobeat`, `blues`, and `middleeastern` were grouped into `others` to reduce extreme class imbalance.

## Feature Extraction

Raw audio files were loaded with `librosa`, resampled to 22,050 Hz, and converted to mono. The following features were extracted:

- Tempo / BPM
- Onset rate
- RMS energy
- Zero-crossing rate
- Spectral centroid
- Spectral bandwidth
- Spectral rolloff
- MFCC means, 13 coefficients
- MFCC standard deviations, 13 coefficients

These features capture both rhythmic behavior and timbral characteristics of the drum recordings.

## Models

Three modeling approaches were compared.

### 1. Gaussian Naive Bayes Baseline

Used as a simple baseline to test whether the extracted features contain useful signal.

Validation accuracy:

- **21.47% top-1 accuracy**

This is above a balanced random baseline for a 14-class task, but the model is limited by its independence and Gaussian assumptions.

### 2. Bayesian Multinomial Logistic Regression / Bayesian Linear Model

Implemented with `PyMC` and trained using ADVI variational inference. The model learns relationships between standardized audio features and style labels while producing posterior distributions instead of only point predictions.

Validation performance across prior settings and hyperparameters:

- **Top-1 accuracy:** approximately **28.83% to 34.97%**
- **Top-2 accuracy:** approximately **42.94% to 55.21%**

Top-2 accuracy is important here because drum styles often overlap. A model may assign high probability to two musically related styles even when the top prediction is not exactly correct.

### 3. Bayesian Neural Network

A Bayesian neural network was used as a nonlinear comparison. It allowed more flexible decision boundaries but was less interpretable than the Bayesian linear model.

Weighted BNN validation performance:

- **Top-1 accuracy:** approximately **21% to 22%**
- **Top-2 accuracy:** approximately **38% to 39%**

In the available experiments, the BNN did not outperform the simpler Bayesian model.

## Key Results

- The task is **14-class drum-only classification**, not binary classification or full-song genre classification.
- Balanced random guessing would be about **7.1%** top-1 accuracy.
- Gaussian Naive Bayes reached **21.47%** validation accuracy.
- The best Bayesian model reached **34.97% top-1 accuracy** and **55.21% top-2 accuracy**.
- Rock was the easiest class to identify in one validation analysis, reaching **70.2%** accuracy.
- Important features included MFCC means, spectral centroid, RMS energy, and onset-related features.

## Interpretation

The results suggest that drum-only style classification is possible but difficult. The model can recover meaningful signal for several major styles, especially rock, punk, latin, funk, and jazz. Minority classes and closely related styles remain difficult because they have fewer samples and overlapping rhythmic vocabulary.

The strict top-1 score should be read in context: this is a 14-class, imbalanced, drum-only classification problem. The stronger result is that Bayesian models improve over the baseline and often place the true class within the top two predictions, which better reflects the ambiguity of genre labels.

The Bayesian approach is useful because it provides uncertainty estimates and interpretable posterior behavior. Instead of only giving a hard label, the model shows when multiple styles are plausible.

## How to Run

1. Open `notebooks/Data_Preprocessing.ipynb` in Google Colab.
2. Run the preprocessing notebook to download the GMD dataset and extract audio features.
3. Save the cleaned feature table as `cleaned.parquet`.
4. Open `notebooks/Music_Genre_Identify_Using_Drum_Pattern.ipynb`.
5. Run the modeling notebook to train and compare the Bayesian models.

The original dataset is large and is not included in this repository. The preprocessing notebook downloads it directly from Google Magenta.

## Tools Used

- Python
- NumPy
- pandas
- scikit-learn
- librosa
- PyMC
- ArviZ
- PyTensor
- Jupyter Notebook / Google Colab
