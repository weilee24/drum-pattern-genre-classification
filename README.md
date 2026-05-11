# Bayesian Modeling of Drum Pattern Styles

This project investigates whether music styles can be classified using **drum-only audio patterns**. Instead of using melody, lyrics, harmony, or full-song production context, the project focuses on rhythmic and spectral features extracted from isolated drum performances.

The project uses the **Groove MIDI Dataset (GMD)** by Google Magenta, which contains human-performed drum recordings with aligned MIDI files and style labels. Audio features are extracted with `librosa`, then used to train and compare Bayesian classification models.

## Project Motivation

Music genre classification is difficult because modern genres often overlap. Many styles share similar production techniques, melodic patterns, and instrumentation. Drum patterns, however, often preserve useful rhythmic and stylistic cues.

The main question of this project is:

> Can musical style be classified using only drum-pattern features extracted from audio?

## Dataset

- Source: Groove MIDI Dataset (GMD), Google Magenta
- Original metadata: 1,150 entries
- Usable audio recordings: 1,090 files
- Final feature table: 1,090 rows and 35 columns
- Final modeling setup: 14 style classes after merging rare styles into an `others` class
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

Three Bayesian-style modeling approaches were compared:

### 1. Gaussian Naive Bayes Baseline

Used as a simple baseline to test whether the extracted features contain useful signal.

Validation accuracy:

- **21.47%**

The result is above random guessing but limited by the model's independence and Gaussian assumptions.

### 2. Bayesian Multinomial Logistic Regression / Bayesian Linear Model

Implemented with `PyMC` and trained using ADVI variational inference. The model learns relationships between standardized audio features and style labels while producing posterior distributions instead of only point estimates.

Validation performance across prior settings and hyperparameters:

- Top-1 accuracy: approximately **28.83% to 34.97%**
- Top-2 accuracy: approximately **42.94% to 55.21%**

The best observed configuration used a uniform prior, 10 hidden units in the tested architecture, and sigma = 0.5.

### 3. Bayesian Neural Network

A Bayesian neural network was used as a nonlinear comparison. It allowed more flexible decision boundaries but was less interpretable than the linear Bayesian model.

Weighted BNN validation performance:

- Top-1 accuracy: approximately **21% to 22%**
- Top-2 accuracy: approximately **38% to 39%**

## Key Results

- The Naive Bayes baseline reached **21.47%** validation accuracy.
- Bayesian models improved performance, reaching roughly **34% to 40%** validation accuracy depending on configuration.
- Top-2 accuracy was substantially higher, suggesting the model often assigns probability mass to plausible related styles even when the top prediction is not exact.
- Rock was the easiest class to identify in the reported per-genre result, reaching **70.2%** accuracy in one validation analysis.
- Important features included MFCC means, spectral centroid, RMS energy, and onset-related features.

## Interpretation

The results suggest that drum-only style classification is possible but difficult. Some styles, especially rock, punk, latin, funk, and jazz, show stronger rhythmic or timbral signatures. Minority classes and closely related genres remain difficult because they have fewer samples and overlapping rhythmic vocabulary.

The Bayesian approach is useful because it provides uncertainty estimates and interpretable posterior behavior. This is important for this task because genre boundaries are naturally ambiguous, especially when only drum audio is available.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── data_description.md
├── results.md
├── model_card.md
├── .gitignore
├── notebooks/
│   ├── Data_Preprocessing.ipynb
│   └── Music_Genre_Identify_Using_Drum_Pattern.ipynb
├── report/
│   └── Bayesian_Modeling_of_Drum_Pattern_Styles.pdf
├── results/
└── data/
```

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

## Notes

This project is designed as a machine learning and Bayesian modeling project. The main contribution is not simply applying an off-the-shelf classifier, but building a full workflow from audio preprocessing to feature extraction, Bayesian modeling, posterior analysis, and error interpretation.
