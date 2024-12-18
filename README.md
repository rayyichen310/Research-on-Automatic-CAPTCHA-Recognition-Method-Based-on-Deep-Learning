# Research on Automatic CAPTCHA Recognition Method Based on Deep Learning
 
## Introduction

This program leverages deep learning technologies (CRNN + GRU) to implement an automatic CAPTCHA recognition workflow. The functionalities include:

### Data Preparation (`prepare`)
Automatically splits the original CAPTCHA images into training, validation, and testing sets, and saves them in `.pt` file format compatible with PyTorch.

### Training OCR Model (`train`)
Trains an OCR model using the CRNN + GRU architecture to automatically recognize the textual content of CAPTCHA images.

### Training Classification Model (`train_classifier`)
When there are multiple training datasets from different sources (e.g., multiple `img-N` folders), a classification model can be trained to identify the source dataset of each CAPTCHA. Subsequently, the corresponding OCR model for that source is used for recognition.

### Evaluating Model (`evaluate`)
Evaluates the trained model using the test set, outputs character-level and sequence-level accuracy, and generates a confusion matrix.

### Prediction (`predict`)
Performs predictions on images in the `predict` folder. The program first uses the classification model to determine which OCR model to apply to each image, then outputs the recognition results to `predict/predictions.csv`.

### Plotting Training Curves (`plot`)
Reads the loss and accuracy records from the training CSV files and plots them as curves using matplotlib and seaborn.

## Environment Requirements (See `requirement.txt` for details)
- Python 3.x
- PyTorch and its dependencies
- torchvision
- Pillow (for image processing)
- numpy, matplotlib, seaborn, scikit-learn (for confusion matrix and data analysis)
- GPU (optional, for accelerated training)

### Experimental Environment
- AMD Ryzen R9 5900HS
- Nvidia RTX 3060 Laptop
- Windows 11

## Project Directory Structure

The default project directory structure is as follows:

```
project_root/
├─ pretrain/
│  ├─ img/                # Pretraining dataset images
│  ├─ model/              # Models storage
│  └─ output/             # Storage for .csv files and confusion matrices
├─ new/
│  ├─ img-1/
│  │  ├─ img/             # First CAPTCHA dataset
│  │  ├─ model/           
│  │  └─ output/
│  ├─ img-2/
│  │  ├─ img/             # Second CAPTCHA dataset
│  │  ...
│  ├─ classifier/         
│  │  ├─ model/           
│  │  └─ output/
│  └─ ...  (Other `img-N` folders)
├─ train-evaluate/
│  ├─ img/                # Used only for performance testing; `train_classifier` does not read this folder!
│  ├─ model/
│  └─ output/
└─ predict/                # Folder for images to be predicted
```

When executing, the program automatically reads from and stores models and data based on the above structure.

## Usage Steps

### 1. Environment Setup
Run `env.ipynb` to download the required datasets and create the necessary directories.

### 2. Data Preparation (`prepare`)
Included in the `train` functionality; no need to execute separately.
- Place the CAPTCHA images to be trained into the corresponding folders (e.g., `pretrain/img` or `new/img-1/img`).
- The image filenames (excluding the extension) represent the CAPTCHA text. For example, `abc1.png` indicates that the CAPTCHA text is `abc1`.
- Run the program and select `prepare` mode:
  ```
  Please enter mode (prepare|train|train_classifier|evaluate|predict|plot): prepare
  ```
- Follow the prompts to select the target folder. The program will automatically split the data into `train`, `valid`, and `test` sets and save them in `.pt` format.

### 3. Training OCR Model (`train`)
- Ensure data preparation is completed.
- Select `train` mode:
  ```
  Please enter mode (prepare|train|train_classifier|evaluate|predict|plot): train
  ```
- The program will ask whether to use pre-trained weights from `pretrain/model/` and which dataset folder to train on.
- You can choose to train with all or a subset of images. (Before using the pre-trained model, first train all CAPTCHAs in the `pretrain` folder!)
- After training, the program saves the best model weights to the corresponding `model` folder and records the training process in a CSV file within the `output` folder.

### 4. Training Classification Model (`train_classifier`)
If there are multiple `img-N` folders, you can first train a classification model to identify the source dataset of each image, then use the corresponding OCR model for recognition.
- Select `train_classifier` mode:
  ```
  Please enter mode (prepare|train|train_classifier|evaluate|predict|plot): train_classifier
  ```
- Follow the prompts to input the number of images to use. The trained classification model will be saved in `new/classifier/model/`.

### 5. Evaluating Model (`evaluate`)
After training, use `evaluate` mode to test the model's performance on the test set:
```
Please enter mode (prepare|train|train_classifier|evaluate|predict|plot): evaluate
```
- Select the corresponding model. The program will output character-level and sequence-level accuracy and generate a confusion matrix.

### 6. Prediction (`predict`)
- Place the images to be recognized in the `predict` folder.
- Run the program and select `predict` mode:
  ```
  Please enter mode (prepare|train|train_classifier|evaluate|predict|plot): predict
  ```
- The program first uses the trained classification model (`new/classifier/model/classifier_model.pt`) to determine which OCR model to use, then performs CAPTCHA recognition. Finally, the results are output to `predict/predictions.csv`.

### 7. Plotting Training Curves (`plot`)
Use `plot` mode to visualize the loss and accuracy curves from the training records in the CSV files:
```
Please enter mode (prepare|train|train_classifier|evaluate|predict|plot): plot
```
- Select the corresponding CSV file to display the loss and accuracy curves.

## Important Notes
- CAPTCHA image filenames should contain only `[a-z 0-9]` characters, and the length should be between 4 to 6 characters. Images not conforming to this specification will be excluded from training.
- For detailed explanations and experimental procedures, please refer to the PDF document.

## Conclusion
This program example provides a complete workflow for CAPTCHA data processing, model training, evaluation, prediction, and result visualization. Users can adjust the code and architecture based on their specific needs to achieve better CAPTCHA recognition performance and customized applications.

## References

- **Yuan, Z.-Y.** (2018). 運用深度神經網絡實現驗證碼識別 [Master's thesis, National Taiwan University]. NTU Repository.

- **Shi, B., Bai, X., & Yao, C.** (2015). An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition. *arXiv preprint* arXiv:1507.05717.

- **Noury, Z., & Rezaei, M.** (2020). Deep-CAPTCHA: A deep learning based CAPTCHA solver for vulnerability assessment. *arXiv preprint* arXiv:2006.08296v2.

-  **qjadud1994.** (2020). CRNN-Keras [Source code]. GitHub. [https://github.com/qjadud1994/CRNN-Keras](https://github.com/qjadud1994/CRNN-Keras)
- **老農的博客.** (2021). 寫給程式設計師的機器學習入門(八) - 卷積神經網路(CNN) - 圖片分類和驗證碼識別. [https://303248153.github.io/ml-08/](https://303248153.github.io/ml-08/)

--- 
---   
  
  

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
│  ├─ img/                # 預訓練資料集的圖片放置處
│  ├─ model/              # 模型存放處
│  └─ output/             # .csv及混淆矩陣存放位置
├─ new/
│  ├─ img-1/
│  │  ├─ img/             # 第一組驗證碼資料集
│  │  ├─ model/           
│  │  └─ output/
│  ├─ img-2/
│  │  ├─ img/             # 第二組驗證碼資料集
│  │  ...
│  ├─ classifier/         
│  │  ├─ model/           
│  │  └─ output/
│  └─ ...  (其他 img-N 資料夾)
├─ train-evaluate/
│  ├─ img/                # 僅用於效能測試，train_classifier不會讀取此資料夾!
│  ├─ model/
│  └─ output/
└─ predict/                # 要進行預測的圖片放置處

```

程式在執行時會依據上述結構，自動從對應的資料夾中讀取與存放模型及資料。

---

## 使用步驟

### 1. 環境建置
- 執行 `env.ipynb` 以下載所需數據集並建立相關資料夾。

### 2. 資料準備 (prepare) 
已包含在train功能中 不須先執行
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
- 驗證碼圖片的檔名中應僅含有 `[a-z 0-9]` 字元，且長度介於 4 到 6 個字元之間。未符合此規範的圖片將不被納入訓練。
- 詳細說明及實驗流程請參考 PDF 文件。

---

## 結語
本程式範例提供了驗證碼資料處理、模型訓練、評估、預測與結果視覺化的完整流程。使用者可視自身需求調整程式碼與架構，以達成更佳的驗證碼辨識效能與客製化應用。

--- 

## 參考資料

1. **Yuan, Z.-Y.** (2018). 運用深度神經網絡實現驗證碼識別 [Master's thesis, National Taiwan University]. NTU Repository.

2. **Shi, B., Bai, X., & Yao, C.** (2015). An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition. *arXiv preprint* arXiv:1507.05717.

3. **Noury, Z., & Rezaei, M.** (2020). Deep-CAPTCHA: A deep learning based CAPTCHA solver for vulnerability assessment. *arXiv preprint* arXiv:2006.08296v2.

4. **qjadud1994.** (2020). CRNN-Keras [Source code]. GitHub. [https://github.com/qjadud1994/CRNN-Keras](https://github.com/qjadud1994/CRNN-Keras)

5. **老農的博客.** (2021). 寫給程式設計師的機器學習入門(八) - 卷積神經網路(CNN) - 圖片分類和驗證碼識別. [https://303248153.github.io/ml-08/](https://303248153.github.io/ml-08/)

