# Bayesian Modeling of Drum Pattern Styles

這個專案是一個 **14-class drum-style classification** 任務，輸入只使用 **drum-only audio**。模型不使用 melody、vocals、lyrics、harmony、artist metadata，也不使用 full-song production context；整個分類只依賴從 isolated drum performances 擷取出的 rhythmic features 與 spectral features。

因為這是 **14 classes** 的分類問題，約 **35% 到 40% top-1 accuracy** 不能用 binary classification 的標準解讀。若是 balanced random guess，top-1 accuracy 大約只有 **7.1%**。再加上 drum styles 本身有高度重疊，且 dataset 有 class imbalance，因此更重要的結果是：Bayesian models 明顯優於 Gaussian Naive Bayes baseline，並達到約 **55% top-2 accuracy**，代表正確 style 經常出現在模型最有信心的前兩個預測之中。

這個專案的目標不是宣稱能完美做 genre recognition，而是測試：在只看 drum patterns 的限制下，模型能從 audio features 中取得多少 style information，並用 Bayesian modeling 分析 prediction uncertainty 與 model limitations。

## Project Motivation

Music genre classification 本身不容易，因為 modern genres 經常混合不同元素。很多 styles 會共享相似的 production techniques、melodic patterns 與 instrumentation。相對來說，drum patterns 往往保留較穩定的 rhythmic cues 與 stylistic cues。

本專案的核心問題是：

> Can musical style be classified using only drum-pattern features extracted from audio?

這個設定是刻意限制過的。用 drum-only audio 做 14-class classification 比用 full songs 做 genre classification 更困難，但它可以更清楚地觀察 rhythmic features 與 timbral features 是否真的包含可用的 style signal。

## Dataset

- Source: Groove MIDI Dataset (GMD), Google Magenta
- Original metadata: 1,150 entries
- Usable audio recordings: 1,090 files
- Final feature table: 1,090 rows and 35 columns
- Final task: **14-class classification**，rare styles 合併成 `others` class
- Split: 70% training, 15% validation, 15% test，使用 stratified sampling

部分樣本數太少的 classes，例如 `dance`、`afrobeat`、`blues`、`middleeastern`，被合併到 `others`，避免 extreme class imbalance 影響 Bayesian inference。

## Feature Extraction

Raw audio 使用 `librosa` 載入，resample 到 22,050 Hz，並轉成 mono。每個 audio clip 擷取以下 features：

- Tempo / BPM
- Onset rate
- RMS energy
- Zero-crossing rate
- Spectral centroid
- Spectral bandwidth
- Spectral rolloff
- MFCC means, 13 coefficients
- MFCC standard deviations, 13 coefficients

這些 features 同時描述 drum recordings 的 rhythmic behavior 與 timbral characteristics。

## Models

本專案比較三種 Bayesian / probabilistic modeling approaches。

### 1. Gaussian Naive Bayes Baseline

Gaussian Naive Bayes 作為 simple baseline，用來檢查 extracted features 是否包含可用的 style signal。

Validation accuracy:

- **21.47% top-1 accuracy**

這個結果高於 14-class task 的 balanced random baseline，但模型受到 feature independence assumption 與 Gaussian assumption 限制，因此表現有限。

### 2. Bayesian Multinomial Logistic Regression / Bayesian Linear Model

主要模型使用 `PyMC` 實作，並透過 ADVI variational inference 近似 posterior distribution。這個模型會學習 standardized audio features 與 style labels 之間的關係，並輸出 posterior probabilities，而不是只有 single hard prediction。

Validation performance across prior settings and hyperparameters:

- **Top-1 accuracy:** 約 **28.83% 到 34.97%**
- **Top-2 accuracy:** 約 **42.94% 到 55.21%**

Top-2 accuracy 在這個專案中特別重要，因為 drum styles 常有 overlap。即使 top-1 prediction 不完全正確，模型仍可能把高 probability 分配給 musically related styles。

### 3. Bayesian Neural Network

Bayesian Neural Network 作為 nonlinear comparison。它能表示更複雜的 decision boundaries，但 interpretability 低於 Bayesian linear model。

Weighted BNN validation performance:

- **Top-1 accuracy:** 約 **21% 到 22%**
- **Top-2 accuracy:** 約 **38% 到 39%**

在目前實驗結果中，BNN 沒有超過較簡單的 Bayesian linear model。

## Key Results

- 任務是 **14-class drum-only classification**，不是 binary classification，也不是 full-song genre classification。
- Balanced random guessing 的 top-1 accuracy 約為 **7.1%**。
- Gaussian Naive Bayes baseline 達到 **21.47%** validation accuracy。
- 最佳 Bayesian model 達到 **34.97% top-1 accuracy** 與 **55.21% top-2 accuracy**。
- 在其中一次 validation analysis 中，rock 是最容易辨識的 class，accuracy 達到 **70.2%**。
- 重要 features 包含 MFCC means、spectral centroid、RMS energy 與 onset-related features。

## Interpretation

結果顯示，drum-only style classification 是可行但困難的任務。模型能從部分 major styles 中抓到有效 signal，特別是 rock、punk、latin、funk、jazz。相對地，minority classes 與 rhythmically similar styles 較難分類，主要原因是樣本數較少且 rhythmic vocabulary 重疊。

Top-1 accuracy 必須放在正確脈絡下看：這是 14-class、imbalanced、drum-only classification problem。比較有意義的結論是，Bayesian models 相比 baseline 有明顯提升，而且經常能把 true class 放進 top-2 predictions，這比較符合 genre labels 本身有 ambiguity 的現實。

Bayesian approach 的價值在於 uncertainty estimation 與 posterior analysis。模型不是只輸出單一 label，而是能顯示多個 styles 同時 plausible 的情況。

## How to Run

1. 用 Google Colab 開啟 `notebooks/Data_Preprocessing.ipynb`。
2. 執行 preprocessing notebook，下載 GMD dataset 並擷取 audio features。
3. 將 cleaned feature table 儲存為 `cleaned.parquet`。
4. 開啟 `notebooks/Music_Genre_Identify_Using_Drum_Pattern.ipynb`。
5. 執行 modeling notebook，訓練並比較 Bayesian models。

原始 dataset 檔案較大，因此不包含在 repository 裡。Preprocessing notebook 會直接從 Google Magenta 下載資料。

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

這個專案的重點不是單純套用 off-the-shelf classifier，而是建立完整 workflow：audio preprocessing、feature extraction、Bayesian modeling、posterior analysis、model comparison 與 error interpretation。

中文版 README 保留主要 technical keywords 的英文寫法，避免在面試中因為專有名詞翻譯不一致造成理解落差。
