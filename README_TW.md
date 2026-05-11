# Bayesian模型:鼓組風格Classification

本專案是一個**14 類鼓組風格分類任務**，僅使用**純鼓組音訊**進行建模。模型不使用旋律、人聲、歌詞、和聲、藝人資訊或完整歌曲的製作脈絡，只依賴從獨立鼓組演奏中提取的節奏與頻譜特徵。

由於任務有**14 個類別**，嚴格的 top-1 Accuracy約在 **35%~40%**。隨機猜測的基準約為 **7.1%**，且許多類別在節奏上高度相似或樣本數較少。更具參考價值的指標是：Bayesian模型明顯優於簡單的 Naive Bayes Baseline，且 top-2 Accuracy可達到約 **55%**，代表正確風格通常出現在模型最可能的前兩個預測中。

## 專案動機

音樂類型分類非常困難，許多風格互相借用相似的製作技術、旋律模式與樂器配置。但鼓組模式通常保留了強烈的節奏與風格特徵。

本專案的核心問題是：

> 是否能**僅使用乾淨的鼓組模式特徵**來分類音樂風格？

由於使用的是乾淨的純鼓組音檔，能有效排除其他聲音的干擾，讓我們更清楚地觀察節奏與音色特徵在風格辨識上所扮演的角色。

## Dataset

- 來源：Groove MIDI Dataset，Google Magenta
- 原始資料：1,150 筆
- 可用音訊檔案：1,090 個
- 最終特徵表格：1,090 列 × 35 欄
- 最終任務：合併稀有類別後的**14 類分類**
- 分割比例：70% Train、15% Test、15% Validation（Stratified）

少數類別（如 `dance`、`afrobeat`、`blues`、`middleeastern`）已合併至 `others` 以降低嚴重的類別不平衡。

## 特徵提取

使用 `librosa` 載入原始音訊，重新取樣至 22,050 Hz 並轉為單聲道。提取以下特徵：

- Tempo / BPM
- Onset rate
- RMS energy
- Zero-crossing rate
- Spectral centroid
- Spectral bandwidth
- Spectral rolloff
- MFCC 平均值（13 個係數）
- MFCC 標準差（13 個係數）

這些特徵同時捕捉了鼓組的節奏行為與音色特性。

## 模型

本專案比較了三種建模方法。

### 1. Gaussian Naive Bayes Baseline

作為簡單基準，用來驗證提取的特徵是否含有有用訊號。

驗證集表現：
- **Top-1 Accuracy：21.47%**

高於 14 類隨機猜測(7.1%)。

### 2. Bayesian Logistic Regression / Bayesian Linear 模型

使用 `PyMC` 與 ADVI 變分推斷實作。模型在學習標準化音訊特徵與風格標籤的關係時，會產生完整的後驗分佈而非單一點估計。

不同Prior設定與超參數下的表現：
- **Top-1 Accuracy：約 28.83% ~ 34.97%**
- **Top-2 Accuracy：約 42.94% ~ 55.21%**

由於鼓組風格常有重疊，top-2 Accuracy在此task中特別有用。即使第一預測錯誤，模型仍常將正確風格列為高度可能的候選之一。

### 3. Bayesian Neural Network

作為非線性模型的對照。雖然具備更彈性的決策邊界，但可解釋性較Bayesian線性模型低。

加權 BNN 驗證表現：
- **Top-1 Accuracy：約 21% ~ 22%**
- **Top-2 Accuracy：約 38% ~ 39%**

在現有實驗中，BNN 並未超越較簡單的Bayesian線性模型且有Overfitting跡象。

## 重要結果

- 這是一個**14 類純鼓組分類任務**，不同於二元分類或完整歌曲類型分類。
- 平衡隨機猜測的 top-1 Baseline約為 **7.1%**。
- Gaussian Naive Bayes 達到 **21.47%** Accuracy。
- 最佳Bayesian模型達到 **34.97%** top-1 與 **55.21%** top-2 Accuracy。
- 重要feature包含：MFCC 平均值、Spectral Centroid、RMS Energy 與 onset 相關的feature。

## 解讀

結果顯示，僅使用鼓組進行風格分類是可行的，但難度較高。模型能為 rock、punk、latin、funk、jazz 等主要風格恢復有意義的訊號。少數類別與高度相似的風格仍較難辨識，這與樣本數少及節奏詞彙重疊有關。

top-1 分數應放在 14-class classification情境下解讀。Bayesian模型大幅優於基準，且多數時候能將正確類別置於前兩名。

## 如何執行

1. 在 Google Colab 中開啟 `Data_Preprocessing.ipynb`。
2. 執行 Data_Preprocessing notebook，下載 GMD 資料集並提取音訊特徵。
3. 將清理後的特徵表存為 `cleaned.parquet`。
4. 開啟 `Music_Genre_Identify_Using_Drum_Pattern.ipynb`。
5. 執行建模 notebook 來訓練與比較Bayesian模型。

原始資料集較大，未包含在 repository 中。前處理 notebook 會直接從 Google Magenta 下載。

## 使用工具

- Python
- NumPy
- pandas
- scikit-learn
- librosa
- PyMC
- ArviZ
- PyTensor
- Jupyter Notebook / Google Colab
