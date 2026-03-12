# File Guessr 📂🔍

一個基於 **本地 AI (Ollama + Gemma 3 4B)** 與 **Elasticsearch** 的強大檔案搜尋工具。
你可以使用自然語言（例如：「紅色的跑車」、「上週的預算報告」）來搜尋電腦中的檔案，不再需要死記硬背檔名。

## ✨ 特色功能

- 🧠 **本地 AI 驅動**: 使用 `gemma3:4b` 模型，所有資料都在本地處理，隱私安全無虞。
- 🔍 **模糊搜尋**: 打錯字也能找到！Elasticsearch 提供強大的模糊匹配。
- 🖼️ **圖片理解**: 自動分析圖片內容並生成描述，讓圖片也能用文字搜尋。
- ⚡ **智能排序**: Elasticsearch 多欄位加權 + BM25 相關性排序。
- 📂 **動態監控**: 自動偵測資料夾變更（新增/修改檔案），即時更新搜尋索引。
- 🚀 **一鍵部署**: 內附 `run.bat` 腳本，Windows 用戶可輕鬆安裝使用。

## 📋 系統需求

- **Windows 10/11**
- **Python 3.10+** (需加入環境變數 PATH)
- **Ollama** (需安裝並執行中)
  - 請至 [ollama.com](https://ollama.com/) 下載安裝
  - 安裝後執行 `ollama pull gemma3:4b` 下載模型
- **Elasticsearch 8.x+** (需安裝並執行中)
  - 請至 [elastic.co](https://www.elastic.co/downloads/elasticsearch) 下載
  - 解壓縮後執行 `bin\elasticsearch.bat` 啟動服務
  - 預設位址: `http://localhost:9200`

## 🚀 快速開始 (Windows)

1. 下載或 Clone 此專案。
2. **啟動 Elasticsearch** (`bin\elasticsearch.bat`)。
3. **啟動 Ollama** (確認它在背景運行)。
4. 雙擊執行 **`run.bat`**。
5. 瀏覽器會自動打開 **`http://127.0.0.1:8000`**。

> 💡 如果沒有安裝 Elasticsearch，搜尋功能會自動降級為 SQLite 模式（無模糊匹配）。

## 🛠️ 手動安裝

```bash
# 1. 建立虛擬環境
python -m venv venv
venv\Scripts\activate

# 2. 安裝相依套件
pip install -r requirements.txt

# 3. 下載 AI 模型
ollama pull gemma3:4b

# 4. 啟動 Elasticsearch
# (在另一個終端) bin\elasticsearch.bat

# 5. 啟動伺服器
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

## 📖 如何使用

1. 點擊右上角的 **⚙️ (設定)** 圖示。
2. 輸入想要索引的資料夾路徑 (例如 `D:\Photos`)。
3. 點擊 **開始索引**。
4. 等待索引完成後，在首頁搜尋框輸入描述即可！

## 🏗️ 系統架構

- **後端**: FastAPI (Python web 框架)
- **搜尋引擎**: Elasticsearch (模糊搜尋 + 相關性排序)
- **資料庫**: SQLite (元數據存儲)
- **AI 模型**: Ollama API (Gemma 3 4B)
- **前端**: Vanilla JS + CSS (玻璃擬態風格 UI)

## 📄 授權

MIT License
