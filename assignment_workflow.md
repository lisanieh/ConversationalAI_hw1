# 作業完成順序備忘

這份作業的主要實作範圍是 `models/crf.py`、`models/bilstm.py`、`train.py` 和 `evaluate.py`。此外還要完成資料量實驗、錯誤分析和提交檔案。

## 1. 確認環境和資料

在 Anaconda Prompt 執行：

```bat
conda activate CAI_SO_FUN
cd /d "C:\Users\AN410\Desktop\研究所\課程\Conversitional AI\hw1\CAI_Assignment_1a"
python data_loader.py
```

`data_loader.py` 會讀取每個資料分割中的 `seq.out`。
如果命令報錯說找不到 `seq.out`，先確認資料是否完整，再開始寫模型。

## 2. 完成評估函式

在 `evaluate.py` 實作 `compute_metrics()`。`scorer.py` 已提供 token-level 和 span-level F1 函式；
實作完成後，訓練時就能用一致的 dev 分數比較模型。

## 3. 先完成 CRF

在 `models/crf.py` 實作 `extract_features()`，
再於 `train.py` 完成 CRF 訓練和 checkpoint 儲存。
用 dev 集調整特徵，並記錄每次嘗試的 F1。

## 4. 再完成 BiLSTM

在 `models/bilstm.py` 實作 `forward()`，然後完成 `train.py` 中 BiLSTM 的訓練、dev 評估和儲存。
checkpoint 要包含模型權重與詞彙／標籤對照，讓 `evaluate.py` 能正確載入並推論。

## 5. 做資料量實驗（Part C）

CRF 和 BiLSTM 都要分別使用 `0.05`、`0.10`、`0.25`、`1.00` 訓練資料。
記錄 dev F1，將兩條曲線畫在同一張圖上，並討論曲線交叉的位置。骨架的模型名稱參數使用小寫 `crf` 和 `bilstm`。

## 6. 做錯誤分析並準備提交（Part D）

在 test 集比較兩個模型的錯誤，挑十個例子分析。
接著產生兩個模型的 test predictions、確認 checkpoint 可以載入，並準備不超過三頁的報告和 AI 使用揭露。

## 容易忽略的細節

- `models/__init__.py` 雖然是空檔，也要保留並一起提交。
- `scorer.py` 和 `data_loader.py` 是提供好的檔案，通常不用修改。
- `train.py` 和 `evaluate.py` 骨架中有預期的 checkpoint 格式；儲存時要遵守，否則訓練可能成功，但評估或重現檢查會失敗。
