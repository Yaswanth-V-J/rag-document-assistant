# RAG Document Assistant

Ask questions about your PDFs and get answers with page-level citations.

**Stack:** Python, Streamlit, ChromaDB's built-in ONNX embeddings, any OpenAI-compatible LLM (Groq default).

## Run
```bash
pip install -r requirements.txt
export LLM_API_KEY=your_key        # free key at console.groq.com
streamlit run app.py
```
Use Ollama locally instead: `LLM_BASE_URL=http://localhost:11434/v1 LLM_API_KEY=ollama LLM_MODEL=llama3.2`

## Pipeline
PDF -> page text -> overlapping chunks -> embeddings -> Chroma -> top-k retrieval -> grounded prompt -> cited answer

## Deploy to Streamlit Community Cloud
1. Push the project to a public GitHub repository. Do not commit `.env`, `.venv`, `chroma_db`, or API keys.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), create a new app and select the repository and `app.py`.
3. In **Advanced settings**, choose Python 3.11 or 3.12.
4. Add this to the app's secrets settings, replacing the placeholder with your own key:

   ```toml
   LLM_API_KEY = "your_groq_api_key"
   ```

The app reads `LLM_API_KEY` from Streamlit secrets when deployed, or from the environment for local use.
