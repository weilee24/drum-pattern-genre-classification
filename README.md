# Bayesian Modeling for Drum Style Classification

This project is a **14-class drum-style classification** task. The goal is to classify the style of each drum pattern using only clean drum audio.

I used Google Magenta's **Groove MIDI Dataset (GMD)** because it contains isolated drum performances recorded by professional drummers. From the audio files, I extracted features related to rhythmic density, onset patterns, dynamics, timbre, and spectral characteristics. The model does not use melody, vocals, or other instruments.

Since this is a **14-class classification** problem, the random top-1 baseline is about **7.1%**. Under this setting, the best Bayesian model reached around **34.97% top-1 accuracy** and **55.21% top-2 accuracy**, clearly outperforming the Gaussian Naive Bayes baseline at **21.47%**. This shows that the model can learn recognizable style patterns from drum audio.

## Motivation

Music genre classification is difficult because many styles borrow similar production techniques, melodic ideas, and instrument arrangements from each other. However, drum patterns often preserve strong rhythmic and stylistic information.

The main question of this project is:

> Can music style be classified using only clean drum pattern features?

By using isolated drum audio, the project focuses directly on rhythm and timbre without interference from other instruments or production layers.

## Dataset

- Source: Groove MIDI Dataset, Google Magenta
- Original samples: 1,150
- Usable audio files: 1,090
- Final feature table: 1,090 rows × 35 columns
- Final task: **14-class classification** after merging rare classes
- Train-test split: 70% Train, 15% Validation, 15% Test, stratified

Rare classes such as `dance`, `afrobeat`, `blues`, and `middleeastern` were merged into `others` to reduce severe class imbalance.

## Feature Extraction

I used `librosa` to load the raw audio, resample it to 22,050 Hz, and convert it to mono. The extracted features include:

- Tempo / BPM
- Onset rate
- RMS energy
- Zero-crossing rate
- Spectral centroid
- Spectral bandwidth
- Spectral rolloff
- MFCC means, 13 coefficients
- MFCC standard deviations, 13 coefficients

These features capture both rhythmic behavior and timbral characteristics of the drum performances.

## Models

This project compares three modeling approaches.

### 1. Gaussian Naive Bayes Baseline

This model serves as a simple baseline to check whether the extracted features contain useful signal.

Validation performance:

- **Top-1 Accuracy: 21.47%**

This is higher than the random baseline of about 7.1% for 14 classes.

### 2. Bayesian Multinomial Logistic Regression

This model was implemented using `PyMC` and ADVI variational inference. Instead of producing only point estimates, the model learns posterior distributions over the parameters while modeling the relationship between standardized audio features and style labels.

Performance across different prior settings and hyperparameters:

- **Top-1 Accuracy: about 28.83% ~ 34.97%**
- **Top-2 Accuracy: about 42.94% ~ 55.21%**

Because many drum styles overlap, top-2 Accuracy helps show whether the model often ranks the correct style near the top.

### 3. Bayesian Neural Network

The Bayesian Neural Network was used as a nonlinear comparison model. It has a more flexible decision boundary, but it is less interpretable than the Bayesian Multinomial Logistic Regression model.

Weighted BNN validation performance:

- **Top-1 Accuracy: about 21% ~ 22%**
- **Top-2 Accuracy: about 38% ~ 39%**

In the current experiments, the BNN did not outperform the simpler Bayesian Multinomial Logistic Regression model and showed signs of overfitting.

## Results

- The random top-1 baseline for 14 balanced classes is about **7.1%**.
- GNB achieved **21.47%** Accuracy as the baseline.
- The best Bayesian model reached **34.97% top-1** and **55.21% top-2 Accuracy**.
- Important features included MFCC means, spectral centroid, RMS energy, and onset-related features.

## Interpretation

The results show that drum-only style classification is possible, but the task is difficult. The model was able to learn recognizable patterns for major styles such as rock, punk, latin, funk, and jazz. Minority classes and highly similar styles remained harder to classify.

In a 14-class classification setting, the Bayesian model clearly outperformed the baseline and often placed the correct class within the top two predictions.

## How to Run

1. Open `Data_Preprocessing.ipynb` in Google Colab.
2. Run the Data_Preprocessing notebook to download the GMD dataset and extract audio features.
3. Save the cleaned feature table as `cleaned.parquet`.
4. Open `Music_Genre_Identify_Using_Drum_Pattern.ipynb`.
5. Run the modeling notebook to train and compare the Bayesian models.

The original dataset is large and is not included in this repository. The preprocessing notebook downloads it directly from Google Magenta.

## Tools

- Python
- NumPy
- pandas
- scikit-learn
- librosa
- PyMC
- ArviZ
- PyTensor
- Jupyter Notebook / Google Colab
