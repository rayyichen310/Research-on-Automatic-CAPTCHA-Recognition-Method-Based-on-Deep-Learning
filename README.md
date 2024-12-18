
# 基於深度學習的驗證碼自動識別方法研究

## 簡介
本程式以深度學習技術（CRNN + GRU）為核心，實現驗證碼自動辨識流程。包含的功能如下：

1. **資料準備 (prepare)**  
   將原始驗證碼圖片自動分割為訓練集、驗證集與測試集，並轉存為 PyTorch 可讀取的 `.pt` 檔案格式。

2. **訓練 OCR 模型 (train)**  
   使用 CRNN + GRU 架構訓練 OCR 模型，以自動辨識驗證碼圖像的文字內容。

3. **訓練分類模型 (train_classifier)**  
   當有多組不同來源的驗證碼訓練資料（例如多個 `img-N` 資料夾）時，可以先訓練分類模型以辨別該驗證碼屬於哪個資料集來源，然後再針對該來源使用對應的 OCR 模型進行辨識。

4. **評估模型 (evaluate)**  
   使用測試集對已訓練的模型進行評估，輸出字元級與序列級的準確率並產生混淆矩陣。

5. **預測 (predict)**  
   對 `predict` 資料夾中的圖片進行預測。程式會先透過分類模型判斷該圖片應使用哪個 OCR 模型，接著輸出辨識結果至 `predict/predictions.csv`。

6. **繪製訓練曲線 (plot)**  
   從訓練紀錄的 CSV 檔案中讀取訓練和驗證集的損失與準確率紀錄，並繪製成曲線圖。

---

## 環境需求  (詳見requirement.txt)
- Python 3.x
- PyTorch 及其相依套件
- torchvision
- Pillow（處理影像）
- numpy、matplotlib、seaborn、scikit-learn（混淆矩陣與數據分析）
- GPU（如有，則可使用加速訓練）

---

## 實驗環境
- AMD Ryzen R9 5900HS
- Nvidia RTX 3060 laptop
- Windows 11
  
---

## 專案目錄結構
專案預設的目錄結構如下：

```
project_root/
├─ pretrain/
│  ├─ img/                # 預訓練資料集圖片
│  ├─ model/              # 預訓練模型存放處
│  └─ output/             # 預訓練過程產生的紀錄與結果
├─ new/
│  ├─ img-1/
│  │  ├─ img/             # 第一組新圖片資料集
│  │  ├─ model/           # 該組訓練後模型存放處
│  │  └─ output/
│  ├─ img-2/
│  │  ├─ img/             # 第二組新圖片資料集
│  │  ...
│  ├─ classifier/         
│  │  ├─ model/           # 分類模型存放處
│  │  └─ output/
│  └─ ...  (其他 img-N 資料夾)
├─ train-evaluate/
│  ├─ img/                (可自行放入想訓練的驗證碼)
│  ├─ model/
│  └─ output/
└─ predict/                # 用於預測的圖片放置處
```

程式在執行時會依據上述結構，自動從對應的資料夾中讀取與存放模型及資料。

---

## 使用步驟

### 1. 環境建置
- 執行 `env.ipynb` 以下載所需數據集並建立相關資料夾。

### 2. 資料準備 (prepare) 
- 已包含在train功能中 不須先執行
- 將待訓練的驗證碼圖片放入對應的資料夾（例如 `pretrain/img` 或 `new/img-1/img`）。
- 圖片檔名（不含副檔名）即為驗證碼文字內容。例如：`abc1.png` 表示該驗證碼文字為 `abc1`。
- 執行程式並選擇 `prepare` 模式：
  ```
  請輸入模式 (prepare|train|train_classifier|evaluate|predict|plot): prepare
  ```
  依照指示選擇目標資料夾後，程式將自動分割為 `train/valid/test` 並存成 `.pt` 格式。

### 3. 訓練 OCR 模型 (train)
- 確保已先執行資料準備後，選擇 `train` 模式：
  ```
  請輸入模式 (prepare|train|train_classifier|evaluate|predict|plot): train
  ```
- 程式會詢問是否使用 `pretrain/model/` 中的預訓練權重，以及選擇哪組資料夾進行訓練。  
  可選擇使用全部或部分圖片進行訓練。（使用預訓練模型前請先訓練 `pretrain` 資料夾中所有驗證碼！）
- 訓練完成後，程式會將最佳模型權重存放至對應的 `model` 資料夾，並在 `output` 資料夾中紀錄訓練過程的 CSV 檔案。

### 4. 訓練分類模型 (train_classifier)
- 若有多組 `img-N` 資料夾，可先透過分類模型辨別每張圖片的來源資料集，然後再使用對應的 OCR 模型。
- 選擇 `train_classifier` 模式：
  ```
  請輸入模式 (prepare|train|train_classifier|evaluate|predict|plot): train_classifier
  ```
- 依程式指示輸入使用的圖片數量，訓練後的分類模型將存放於 `new/classifier/model` 資料夾中。

### 5. 評估模型 (evaluate)
- 完成訓練後，可使用 `evaluate` 模式測試模型在測試集的效果：
  ```
  請輸入模式 (prepare|train|train_classifier|evaluate|predict|plot): evaluate
  ```
- 選擇對應的模型後，將產生字元級與序列級準確率、並輸出混淆矩陣。

### 6. 預測 (predict)
- 將欲辨識的圖片放入 `predict` 資料夾，然後執行：
  ```
  請輸入模式 (prepare|train|train_classifier|evaluate|predict|plot): predict
  ```
- 程式先用已訓練好的分類模型 (`new/classifier/model/classifier_model.pt`) 辨別應使用的 OCR 模型，接著進行驗證碼辨識。最後將結果輸出至 `predict/predictions.csv`。

### 7. 繪製訓練曲線 (plot)
- 使用 `plot` 模式可從訓練紀錄的 CSV 檔案中繪製損失及準確率曲線：
  ```
  請輸入模式 (prepare|train|train_classifier|evaluate|predict|plot): plot
  ```
- 選擇對應的 CSV 檔案後，即可顯示損失與準確率曲線圖。

---

## 注意事項
- 驗證碼圖片的檔名中應僅含有 `[a-z0-9]` 字元，且長度介於 4 到 6 個字元之間。未符合此規範的圖片將不被納入訓練。
- 詳細說明及實驗流程請參考 PDF 文件。

---

## 結語
本程式範例提供了驗證碼資料處理、模型訓練、評估、預測與結果視覺化的完整流程。使用者可視自身需求調整程式碼與架構，以達成更佳的驗證碼辨識效能與客製化應用。

--- 

## 參考資料

1. Yuan, Z.-Y. (2018). 運用深度神經網絡實現驗證碼識別 [Master's thesis, National Taiwan University]. NTU Repository.
2. Shi, B., Bai, X., & Yao, C. (2015). An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition. arXiv preprint arXiv:1507.05717.
3. Noury, Z., & Rezaei, M. (2020). Deep-CAPTCHA: A deep learning based CAPTCHA solver for vulnerability assessment. arXiv preprint arXiv:2006.08296v2.
4. qjadud1994. (2020). CRNN-Keras [Source code]. GitHub. [https://github.com/qjadud1994/CRNN-Keras](https://github.com/qjadud1994/CRNN-Keras)
5. 老農的博客. (2021). 寫給程式設計師的機器學習入門(八) - 卷積神經網路(CNN) - 圖片分類和驗證碼識別. [https://303248153.github.io/ml-08/](https://303248153.github.io/ml-08/)

