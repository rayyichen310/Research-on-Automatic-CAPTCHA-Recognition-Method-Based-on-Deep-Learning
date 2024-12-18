# Research-on-Automatic-CAPTCHA-Recognition-Methods-Based-on-Deep-Learning-Abstract

# 基於深度學習的驗證碼自動識別方法研究

## 簡介

此程式旨在透過深度學習技術（基於 CRNN + GRU）實現簡單的驗證碼辨識流程。程式中包含以下功能：

1. **準備資料 (prepare)**：將原始圖片分割成訓練集、驗證集與測試集，並存成 PyTorch 可讀取的格式 ( `.pt` )。
2. **訓練模型 (train)**：使用 CRNN + GRU 架構訓練 OCR 模型，用於辨識驗證碼內容。
3. **訓練分類模型 (train_classifier)**：當有多組不同來源的驗證碼訓練集（如多個 `img-N` 資料夾）時，可先訓練一個分類模型來分類輸入的驗證碼圖片屬於哪個來源，接著再選擇對應的 OCR 模型進行辨識。
4. **評估模型 (evaluate)**：對測試集進行評估，顯示字符級與序列級的準確率，並產出混淆矩陣。
5. **預測 (predict)**：對 `predict` 資料夾中的所有圖片進行預測，程式將先透過分類模型判斷該圖片應使用哪個 OCR 模型，再進行驗證碼辨識。
6. **繪製訓練曲線 (plot)**：從訓練過程中紀錄下的 CSV 檔案中，讀取訓練與驗證的損失及準確率紀錄，並繪製出曲線圖。

## 環境需求

- Python 3.x
- PyTorch 及其相依套件
- torchvision (已使用部分模型與功能)
- Pillow (處理影像)
- numpy、matplotlib、seaborn、scikit-learn (混淆矩陣與數據分析)
- GPU（如有，則可使用加速訓練）

## 專案目錄結構

假設專案的目錄結構如下：

```
project_root/
├─ pretrain/
│  ├─ img/                # 預訓練資料集的圖片放置處
│  ├─ model/              # 預訓練模型存放處
│  └─ output/             # 預訓練過程中產出的紀錄與結果
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
│  ├─ img/                # 用於最終評估訓練結果的圖片資料集
│  ├─ model/
│  └─ output/
└─ predict/                # 要進行預測的圖片放置處
```

程式在執行時會依賴上述結構，並自動在對應資料夾內讀取或存放模型、資料。

## 使用步驟



1. **資料準備 (prepare)**
   執行evn.ipynb 下載數據集並建立相關資料夾

2. **資料準備 (prepare)**

   請先將欲訓練之驗證碼圖片放入對應的資料夾中 (例如 `pretrain/img` 或 `new/img-1/img`)。  
   圖片名稱的檔名（去掉副檔名後）即代表該圖片的驗證碼文字，例如 `abc1.png` 表示驗證碼為 `abc1`。

   執行程式後輸入：
   ```
   請輸入模式 (prepare|train|train_classifier|evaluate|predict|plot): prepare
   ```
   接著依照指示選擇要準備的資料夾，即可自動分割為 train/valid/test 並存成 `.pt` 格式。

3. **訓練 OCR 模型 (train)**

   在確保已先執行資料準備後，選擇 `train` 模式：
   ```
   請輸入模式 (prepare|train|train_classifier|evaluate|predict|plot): train
   ```
   接著程式會詢問是否使用預訓練模型 (`pretrain/model/`) 中的權重開始訓練，以及選擇哪一組資料夾來訓練，並可選擇使用全部或部分圖片進行訓練。  
   完成後，程式將自動進行訓練並記錄過程至 CSV 檔案，同時在對應的 `model` 資料夾中存放最佳的模型權重。

4. **訓練分類模型 (train_classifier)**

   若您有多組 `img-N` 類型的資料夾，可以先透過分類模型來學習如何分辨圖片來源。此步驟會要求您輸入要從各資料夾中選擇多少圖片進行訓練。
   執行：
   ```
   請輸入模式 (prepare|train|train_classifier|evaluate|predict|plot): train_classifier
   ```
   根據程式指示來輸入圖片數量。訓練完成後，分類模型將被儲存在 `new/classifier/model` 資料夾中。

5. **評估模型 (evaluate)**

   當您完成訓練後，可使用 `evaluate` 模式選擇一個資料夾與該資料夾中訓練而來的模型權重，並對測試集進行評估。
   執行：
   ```
   請輸入模式 (prepare|train|train_classifier|evaluate|predict|plot): evaluate
   ```
   選擇對應的模型，即可產生字元與序列級的準確率以及混淆矩陣。

6. **預測 (predict)**

   將欲預測的圖片放入 `predict` 資料夾中，然後執行：
   ```
   請輸入模式 (prepare|train|train_classifier|evaluate|predict|plot): predict
   ```
   程式會先使用已訓練好的分類模型（`new/classifier/model/classifier_model.pt`）來判斷該張圖片應使用哪個 `img-N` OCR 模型，然後再進行驗證碼辨識。最後將結果輸出至 `predict/predictions.csv`。

7. **繪製訓練曲線 (plot)**

   訓練過程產生的 CSV 檔案會記錄每個 epoch 的訓練與驗證損失、準確率。您可以使用：
   ```
   請輸入模式 (prepare|train|train_classifier|evaluate|predict|plot): plot
   ```
   接著選擇對應的 CSV 檔案，即可顯示對應的損失及準確率曲線圖。

## 注意事項

- 請確保驗證碼圖片的檔名中僅含有 `[a-z0-9]` 且長度介於 4 到 6 個字元之間，否則將不會被納入訓練。
- 資料準備後會自動將訓練、驗證與測試資料分別存成 `.pt` 檔案，未來可以直接使用，不須重複運行 `prepare`。
- 若需要使用 GPU 加速，請確保安裝對應的 CUDA 版本及 GPU driver，程式會自動偵測 `torch.cuda.is_available()`。
- 若需修改影像大小或模型結構，可修改程式中對應的常數與模組（如 `IMAGE_SIZE`、`CRNN_GRU` 等）。

## 結語

本程式提供驗證碼資料處理、模型訓練與預測的完整流程範例。使用者可依實際需求對程式碼與架構進行調整。
