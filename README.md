# 🩺 Sanjeevani AI 2.0

AI Medical Assistant built using LangChain, FAISS, Groq, and Streamlit.

## Features
- 💬 Text Chat — RAG pipeline over medical PDFs (Gale Encyclopedia, MBBS textbook)
- 🧠 RAG + FAISS — semantic search over indexed documents
- 👁️ Vision — attach a photo (e.g. skin rash) and get an AI analysis
- 🎤 Voice Input — ask by speaking, transcribed and sent to the same chatbot
- 🔊 Voice Output — every answer is read aloud
- 📄 Medical Document Upload — add your own PDFs to the knowledge base at runtime
- 📚 Source-based Answers — see which document/page an answer came from
- 🔐 Login / Register — simple local accounts (SQLite, no external service)
- 💬 Chat History — past conversations saved per user, resume anytime
- 🛡️ Medical Safety Layer — blocks dangerous queries, adds a disclaimer to every answer

## Tech Stack
Python, Streamlit, LangChain, FAISS, HuggingFace, Groq (Whisper + Llama-4 Vision + LLM), gTTS, SQLite

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project root with:
   ```
   GROQ_API_KEY=your_groq_key_here
   HF_TOKEN=your_huggingface_token_here
   ```
   (Groq API key is free — get it at https://console.groq.com)

3. Build the vector store (only needed once, or after adding new PDFs to `data/`):
   ```
   python phase1.py
   ```

4. Run the app:
   ```
   streamlit run phase3.py
   ```

5. Register a new account from the app, then log in.

## Notes
- All chat data is stored locally in `sanjeevani.db` (SQLite) — nothing leaves your machine except API calls to Groq/HuggingFace.
- Uploaded PDFs are saved to `data/user_uploads/` and merged into the shared vector store.
