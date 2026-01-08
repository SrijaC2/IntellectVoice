# 🎧 IntellectVoice – AI-Powered PDF to Audiobook & Knowledge Assistant

**IntellectVoice** is a Generative AI–driven web application that transforms PDFs into **audiobooks**, while also enabling **summarization, paraphrasing, keyword extraction, and interactive Q&A** with document content — making information more accessible and engaging.

---

## 🚀 Key Features

- 📄 Upload and process PDF documents
- 🔊 Convert PDF content into **audio (MP3)** using Text-to-Speech
- 🧠 **AI-powered summarization** of document content
- ✍️ **Paraphrasing** using Pegasus Transformer models
- 🔑 **Keyword extraction** for quick insights
- 💬 Chat with PDFs using **semantic search + LLMs**
- ⚡ Fast similarity search using **FAISS vector database**
- 🌐 Web interface built with **Flask**

---

## 🧠 How It Works

1. PDF is uploaded and split into pages
2. Text is converted into embeddings using **HuggingFace models**
3. Embeddings are stored in **FAISS** for semantic search
4. User actions trigger:
   - Summarization via **Gemini Pro**
   - Paraphrasing via **Pegasus Transformer**
   - Keyword extraction using LLM prompting
   - Question answering via contextual retrieval
5. Text-to-Speech converts AI output into downloadable audio

---

## 🛠️ Tech Stack

### Backend & AI
- Python
- Flask
- LangChain
- Google Gemini Pro (LLM)
- HuggingFace Transformers
- Pegasus Paraphrasing Model
- FAISS Vector Store
- Sentence Splitter

### NLP & Audio
- HuggingFace Embeddings (`all-MiniLM-L6-v2`)
- gTTS (Google Text-to-Speech)

---

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/SrijaC2/IntellectVoice.git
cd IntellectVoice
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a .env file:
```bash
GEMINI_API_KEY=your_google_gemini_api_key
```

### 4. Run the Application
```bash
python app.py
```
