# 🧭 CodeCite — Ask Your Repository

CodeCite is a Retrieval-Augmented Generation (RAG) tool that lets you ask natural-language questions about any codebase — local or on GitHub — and get answers grounded in the **exact file and line numbers** where the relevant code lives. It also generates a first-draft `README.md` section directly from the structure of the indexed repository.

Built with **LangChain**, **ChromaDB**, **Streamlit**, and **Google Gemini**.

---

## ✨ Features

- **Index any repository** — point it at a local folder path or a public GitHub URL.
- **Language-aware chunking** — uses LangChain's `RecursiveCharacterTextSplitter.from_language` so Python, JS/TS, Java, Kotlin, Go, Rust, C/C++, C#, PHP, Ruby, Swift, Scala, Markdown, and HTML are split along syntactically meaningful boundaries instead of arbitrary character counts.
- **File-and-line grounded citations** — every retrieved chunk is mapped back to its exact `(start_line, end_line)` in the source file, so answers can cite precisely where code lives instead of just which file.
- **Strictly grounded Q&A** — the LLM is instructed to answer only from retrieved context, and to say plainly when something isn't found rather than fabricate a file path or function.
- **Two embedding options** — run fully offline and free with local `sentence-transformers`, or use Gemini embeddings for a hosted alternative.
- **Automatic rate-limit handling** — batches embedding calls and backs off with exponential retry on transient `429` errors (common on free-tier embedding APIs).
- **README generator** — builds a directory tree + regex-extracted function/class signatures from the indexed repo and asks Gemini to draft an `Overview`, `Project Structure`, and `Key Components` section from it — without dumping raw file contents into the prompt.
- **Chat-style UI** — persistent chat history per session, with expandable source-chunk citations under each answer.

---

## 🏗️ Project Structure

```
.
├── app.py                # Streamlit front-end: sidebar indexing controls, chat tab, README tab
├── indexer.py             # Repo resolution, file walking, chunking, embedding, Chroma vector store
├── qa_chain.py            # Retrieval + grounded-answer prompt chain
└── readme_generator.py    # Structural repo summary + README drafting chain
```

| File | Responsibility |
|---|---|
| `indexer.py` | Clones/resolves the repo, walks source files (skipping `.git`, `node_modules`, build artifacts, etc.), splits each file into language-aware chunks with accurate line-number metadata, embeds them, and persists them to a per-repo Chroma collection. |
| `qa_chain.py` | Retrieves the top-`k` relevant chunks for a question and asks Gemini to answer **only** from that context, citing file paths and line numbers. |
| `readme_generator.py` | Builds a lightweight directory-tree + signature summary of the repo (no raw file dumps) and asks Gemini to draft a README section from it. |
| `app.py` | Streamlit UI tying it all together — indexing controls, chat interface, and README generation tab. |

---

## ⚙️ How It Works

1. **Resolve** — `resolve_source()` accepts a local directory path or a GitHub URL (cloned via GitPython with `depth=1`).
2. **Chunk** — `build_documents()` walks all indexable files and splits them using a language-specific `RecursiveCharacterTextSplitter`, computing exact line ranges for every chunk by locating it in the original file text.
3. **Embed & Store** — chunks are embedded (locally via `sentence-transformers/all-MiniLM-L6-v2`, or via Gemini's `gemini-embedding-001`) and stored in a Chroma vector store, unique per repository source (keyed by a SHA-1 hash of the source string).
4. **Retrieve & Answer** — `answer_question()` retrieves the top-`k` most relevant chunks and passes them to Gemini (`gemini-2.5-flash`) with a strict system prompt: answer only from context, cite file/line, and admit when something isn't found.
5. **Generate README** — `generate_readme_section()` builds a directory tree and extracts top-level function/class names per file via regex, then asks Gemini to draft a plain-language `README.md` section from that structural summary alone.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A [Google Gemini API key](https://aistudio.google.com/app/apikey) (free tier available)
- Git (for indexing GitHub URLs)

### Installation

```bash
git clone <this-repo-url>
cd codecite
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

### Run

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Usage

1. Enter your Gemini API key in the sidebar (if not set via `.env`).
2. Enter a local folder path or a GitHub URL under **"Local folder path OR GitHub URL"**.
3. Choose an embedding backend — **Local** (free, offline, recommended) or **Gemini** (hosted, subject to free-tier rate limits).
4. Click **📥 Index repository** and wait for chunking + embedding to complete.
5. Switch to the **💬 Ask the repo** tab and ask questions like:
   - *"Where is user authentication implemented?"*
   - *"How does the rate-limit retry logic work?"*
6. Switch to the **📄 Generate README** tab to draft a README section from the indexed repo's structure.

---

## 🧩 Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Orchestration | LangChain (`langchain-core`, `langchain-text-splitters`, `langchain-google-genai`) |
| Vector Store | ChromaDB (`langchain-chroma`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) or Gemini `gemini-embedding-001` |
| LLM | Google Gemini (`gemini-2.5-flash`) |
| Repo Access | GitPython |

---

## ⚠️ Known Limitations

- Line-number attribution for a chunk relies on locating its text within the original file; files with large amounts of duplicated content (e.g. repeated license headers) can occasionally cause a chunk to be attributed to the first matching occurrence rather than its true position.
- The README generator's signature extraction currently covers Python, JavaScript, TypeScript, Java, and Go; other indexed languages (C++, Rust, Kotlin, Swift, C#, PHP, Ruby, Scala) will still appear in the directory tree but won't contribute to the "Key Components" list.
- Each Chroma vector store is deleted and rebuilt from scratch on every index run — there's no incremental re-indexing of only changed files yet.
- Large repositories on the free Gemini embedding tier (100 requests/minute) will index more slowly due to automatic rate-limit backoff; local embeddings are recommended for bigger codebases.

---

## 📄 License

This project was built as part of an academic coursework submission. Add your preferred license here (e.g. MIT) if distributing publicly.