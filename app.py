"""
app.py
CodeCite: Ask Your Repository -- Streamlit Front-End (Interactive & Professional Edition)

Run with: streamlit run app.py
"""

import os
import time
import json
import datetime
from collections import Counter
import streamlit as st
from dotenv import load_dotenv

from indexer import (
    resolve_source, build_documents, get_embeddings,
    build_vectorstore, load_vectorstore, get_index_workdir, get_persist_dir_for,
)
from qa_chain import answer_question
from readme_generator import generate_readme_section, build_repo_summary

load_dotenv()

# Securely load Google API key from environment (.env) - user does not enter it in app
google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()

# ------------------------------------------------------------------ API Quota & Limit Tracker --
API_USAGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".api_usage.json")
MAX_API_LIMIT = int(os.getenv("MAX_API_LIMIT", "50"))

def get_api_usage():
    """Returns (used_count, max_limit) for today."""
    today_str = datetime.date.today().isoformat()
    if os.path.exists(API_USAGE_FILE):
        try:
            with open(API_USAGE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("date") == today_str:
                return data.get("calls_used", 0), data.get("max_calls", MAX_API_LIMIT)
        except Exception:
            pass
    return 0, MAX_API_LIMIT

def increment_api_usage(count: int = 1) -> bool:
    """Increments API usage count if within daily limit. Returns True if allowed, False if exceeded."""
    used, limit = get_api_usage()
    if used + count > limit:
        return False
    today_str = datetime.date.today().isoformat()
    new_used = used + count
    try:
        with open(API_USAGE_FILE, "w", encoding="utf-8") as f:
            json.dump({"date": today_str, "calls_used": new_used, "max_calls": limit}, f, indent=2)
    except Exception:
        pass
    return True

def check_api_limit(needed: int = 1) -> bool:
    """Checks if `needed` API calls are within the remaining daily limit."""
    used, limit = get_api_usage()
    return (used + needed) <= limit

def reset_api_usage():
    """Resets the API usage counter for testing or new evaluation session."""
    today_str = datetime.date.today().isoformat()
    try:
        with open(API_USAGE_FILE, "w", encoding="utf-8") as f:
            json.dump({"date": today_str, "calls_used": 0, "max_calls": MAX_API_LIMIT}, f, indent=2)
    except Exception:
        pass


# ---------------------------------------------------------------- Page Config --
st.set_page_config(
    page_title="CodeCite AI | Ask Your Repository",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------- Custom CSS --
st.markdown("""
<style>
    /* Global Typography & Background Tweaks */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    code, pre, .stCodeBlock {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Hero Banner Styling */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
    }
    
    .hero-title {
        font-size: 28px;
        font-weight: 700;
        background: linear-gradient(90deg, #60A5FA, #A855F7, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .hero-subtitle {
        color: #94A3B8;
        font-size: 15px;
        margin-top: 4px;
    }

    /* Status Pills */
    .status-pill-ready {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 13px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    
    .status-pill-waiting {
        background: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 13px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    /* Metric Cards Custom Styling */
    div[data-testid="metric-container"] {
        background-color: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 16px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        border-color: rgba(96, 165, 250, 0.4);
    }

    /* Custom Badges for Source Citations */
    .badge-path {
        background: rgba(59, 130, 246, 0.15);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.3);
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-lines {
        background: rgba(168, 85, 247, 0.15);
        color: #C084FC;
        border: 1px solid rgba(168, 85, 247, 0.3);
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        margin-left: 8px;
    }
    .badge-lang {
        background: rgba(236, 72, 153, 0.15);
        color: #F472B6;
        border: 1px solid rgba(236, 72, 153, 0.3);
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        margin-left: 8px;
        text-transform: uppercase;
    }

    /* Interactive Starter Chip Label */
    .chip-header {
        font-size: 14px;
        font-weight: 600;
        color: #E2E8F0;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    /* Streamlit Tabs Polish */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre;
        border-radius: 8px;
        padding: 0 20px;
        font-weight: 500;
        color: #94A3B8;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(59, 130, 246, 0.2) !important;
        color: #60A5FA !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- Session State --
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "repo_path" not in st.session_state:
    st.session_state.repo_path = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {question, answer, sources}
if "repo_stats" not in st.session_state:
    st.session_state.repo_stats = None
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "readme_output" not in st.session_state:
    st.session_state.readme_output = None
if "source_input_val" not in st.session_state:
    st.session_state.source_input_val = ""


def compute_repo_stats(documents, repo_path):
    file_paths = set()
    lang_counts = Counter()
    for d in documents:
        fp = d.metadata.get("file_path", "")
        if fp:
            file_paths.add(fp)
        lang = d.metadata.get("language", "unknown")
        if lang:
            lang_counts[lang] += 1
            
    top_lang = lang_counts.most_common(1)[0][0].upper() if lang_counts else "N/A"
    return {
        "total_chunks": len(documents),
        "total_files": len(file_paths),
        "top_language": top_lang,
        "language_breakdown": dict(lang_counts.most_common(6)),
        "repo_name": os.path.basename(os.path.abspath(repo_path)) if repo_path else "Repository",
    }


# ------------------------------------------------------------------ Sidebar --
with st.sidebar:
    st.title("🧭 CodeCite AI")
    st.caption("Ground your questions directly in your repository source files.")
    st.divider()

    # Section 1: API & Quota Status
    with st.expander("⚡ API & Quota Status", expanded=True):
        if google_api_key:
            st.markdown("<span style='color: #34D399; font-size: 13px; font-weight: 500;'>✓ System Gemini API pre-configured (.env)</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color: #F87171; font-size: 13px; font-weight: 500;'>⚠️ GOOGLE_API_KEY missing from .env</span>", unsafe_allow_html=True)

        used, limit = get_api_usage()
        remaining = max(0, limit - used)
        percentage = min(1.0, used / max(1, limit))
        
        st.write("")
        st.progress(percentage, text=f"📊 Daily API Budget: {remaining} / {limit} left")
        st.caption(f"**Usage:** `{used}/{limit}` requests used today. Rate limiting prevents free tier exhaustion during project evaluation.")
        
        if st.button("🔄 Reset Quota Counter", use_container_width=True, help="Reset daily API usage counter"):
            reset_api_usage()
            st.rerun()

        st.divider()
        embedding_choice = st.radio(
            "Embedding Model",
            ["Local (free, sentence-transformers)", "Gemini (needs API key)"],
            index=0,
            help="Local runs 100% offline and fast with no rate limits. Gemini uses API quota."
        )
        use_gemini_embeddings = embedding_choice.startswith("Gemini")

    # Section 2: Indexing Controls
    with st.expander("📥 Repository Indexer", expanded=True):
        source = st.text_input(
            "Local folder path OR GitHub URL",
            value=st.session_state.source_input_val,
            placeholder="/path/to/project  or  https://github.com/user/repo"
        )
        
        # Interactive Quick Fill Buttons
        st.write("<span style='font-size: 12px; color: #94A3B8;'>⚡ Quick Fill Helpers:</span>", unsafe_allow_html=True)
        col_q1, col_q2 = st.columns(2)
        with col_q1:
            if st.button("Current Folder (.)", use_container_width=True, help="Fill current workspace directory"):
                st.session_state.source_input_val = "."
                st.rerun()
        with col_q2:
            if st.button("Clear Input", use_container_width=True):
                st.session_state.source_input_val = ""
                st.rerun()

        if st.button("🚀 Index Repository", type="primary", use_container_width=True):
            if not source:
                st.error("Please enter a local folder path or GitHub URL.")
            elif use_gemini_embeddings and not google_api_key:
                st.error("Gemini embeddings require a valid API key pre-configured in .env.")
            elif use_gemini_embeddings and not check_api_limit(1):
                used, limit = get_api_usage()
                st.error(f"⚠️ Daily API budget reached ({used}/{limit} requests used). Please switch to Local (free) embeddings or click 'Reset Quota Counter' in the sidebar.")
            else:
                with st.spinner("Resolving repository structure..."):
                    try:
                        repo_path = resolve_source(source, get_index_workdir())
                    except Exception as e:
                        st.error(f"Could not load repository: {e}")
                        repo_path = None

                if repo_path:
                    progress_bar = st.progress(0, text="Walking repository source files...")

                    def _progress(done, total, current_file):
                        progress_bar.progress(
                            done / max(total, 1),
                            text=f"Chunking `{current_file}` ({done}/{total})"
                        )

                    with st.spinner("Splitting code into exact line-numbered chunks..."):
                        documents = build_documents(repo_path, progress_callback=_progress)

                    if not documents:
                        st.error("No indexable source files found in that directory.")
                    else:
                        with st.spinner(f"Generating embeddings for {len(documents)} chunks..."):
                            embeddings = get_embeddings(use_gemini_embeddings, google_api_key)
                            persist_dir = get_persist_dir_for(source)
                            vectorstore = build_vectorstore(documents, persist_dir, embeddings)
                            if use_gemini_embeddings:
                                increment_api_usage(max(1, len(documents) // 10))

                        st.session_state.vectorstore = vectorstore
                        st.session_state.repo_path = repo_path
                        st.session_state.repo_stats = compute_repo_stats(documents, repo_path)
                        st.session_state.chat_history = []
                        st.session_state.readme_output = None
                        progress_bar.empty()
                        st.success(f"Indexed {len(documents)} chunks successfully!")
                        st.rerun()

    # Section 3: Active Session Info & Reset
    if st.session_state.repo_path:
        st.divider()
        st.caption(f"📍 Active Index Path: `{st.session_state.repo_path}`")
        if st.button("🗑️ Clear & Reset Index", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.repo_path = None
            st.session_state.repo_stats = None
            st.session_state.chat_history = []
            st.session_state.readme_output = None
            st.rerun()

# ------------------------------------------------------------------ Hero Header --
hero_status_class = "status-pill-ready" if st.session_state.vectorstore else "status-pill-waiting"
hero_status_text = f"🟢 Ready — {st.session_state.repo_stats['repo_name']}" if st.session_state.repo_stats else "🔴 Waiting for Indexing..."

st.markdown(f"""
<div class="hero-container">
    <div>
        <h1 class="hero-title">CodeCite Repository Intelligence</h1>
        <div class="hero-subtitle">Grounded, file-and-line cited codebase exploration and automated documentation studio.</div>
    </div>
    <div class="{hero_status_class}">
        {hero_status_text}
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------- KPI Metrics Grid --
if st.session_state.repo_stats:
    stats = st.session_state.repo_stats
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("📂 Indexed Repository", stats["repo_name"])
    with c2:
        st.metric("📄 Source Files", f"{stats['total_files']:,}")
    with c3:
        st.metric("🧩 Vector Chunks", f"{stats['total_chunks']:,}")
    with c4:
        st.metric("💻 Primary Language", stats["top_language"])
    st.write("")  # spacing

# ----------------------------------------------------------------------- Tabs --
tab_chat, tab_explorer, tab_readme = st.tabs([
    "💬 Ask Codebase AI",
    "🌳 Structure & Signatures",
    "📄 Documentation Studio"
])

# ==============================================================================
# TAB 1: INTERACTIVE Q&A CHAT
# ==============================================================================
with tab_chat:
    if not st.session_state.vectorstore:
        st.info("👋 Welcome! Please index a local folder path (like `.`) or GitHub repository from the sidebar to start asking questions.")
    else:
        # Interactive Quick Starters
        st.markdown('<div class="chip-header">💡 Quick Starter Questions (Click to Ask):</div>', unsafe_allow_html=True)
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            if st.button("🏗️ Architecture & Entrypoint", use_container_width=True):
                st.session_state.pending_question = "Where is the main application logic and entrypoint implemented?"
                st.rerun()
        with col_s2:
            if st.button("🔐 API & Security Handling", use_container_width=True):
                st.session_state.pending_question = "Where and how are API keys, environment variables, and authentication handled?"
                st.rerun()
        with col_s3:
            if st.button("⚙️ Indexing & Splitting Logic", use_container_width=True):
                st.session_state.pending_question = "Explain how repository files are chunked, split, and embedded into the vector store."
                st.rerun()
        with col_s4:
            if st.button("📦 Dependencies & Run Instructions", use_container_width=True):
                st.session_state.pending_question = "What external libraries and packages are required, and how is the app launched?"
                st.rerun()

        st.divider()

        # Chat History Action Bar
        if st.session_state.chat_history:
            c_hist1, c_hist2 = st.columns([8, 2])
            with c_hist1:
                st.caption(f"Showing **{len(st.session_state.chat_history)}** conversation turn(s). All answers are grounded strictly in retrieved code chunks.")
            with c_hist2:
                if st.button("🗑️ Clear Chat History", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()

        # Render Previous Turns
        for idx, turn in enumerate(st.session_state.chat_history):
            with st.chat_message("user", avatar="🧑‍💻"):
                st.write(turn["question"])
            with st.chat_message("assistant", avatar="🧭"):
                st.write(turn["answer"])
                
                if turn["sources"]:
                    with st.expander(f"📎 Grounded Citations ({len(turn['sources'])} Retrieved Source Chunks)", expanded=False):
                        tab_labels = [f"📄 Chunk {s_idx+1}: {src['file_path']}" for s_idx, src in enumerate(turn["sources"])]
                        source_tabs = st.tabs(tab_labels)
                        
                        for s_idx, (src_tab, src) in enumerate(zip(source_tabs, turn["sources"])):
                            with src_tab:
                                loc = src["file_path"]
                                lines_badge = f"Lines {src['start_line']} - {src['end_line']}" if src["start_line"] else "Entire File"
                                lang = src.get("file_path", "").split(".")[-1]
                                
                                st.markdown(f"""
                                <div style="margin-bottom: 12px; display: flex; align-items: center; flex-wrap: wrap;">
                                    <span class="badge-path">{loc}</span>
                                    <span class="badge-lines">{lines_badge}</span>
                                    <span class="badge-lang">{lang}</span>
                                </div>
                                """, unsafe_allow_html=True)
                                st.code(src["snippet"], language=lang if lang in ["py", "js", "ts", "java", "go", "cpp", "html", "md"] else None)

        # Handle Chat Input & Pending Questions
        user_input = st.chat_input("Ask about functions, classes, features, or architecture...")
        question_to_process = user_input or st.session_state.pending_question
        
        if question_to_process:
            st.session_state.pending_question = None  # reset pending
            if not google_api_key:
                st.error("⚠️ Gemini API key not configured in environment (.env file).")
            elif not check_api_limit(1):
                used, limit = get_api_usage()
                st.error(f"⚠️ Daily API budget reached ({used}/{limit} requests used). Please click 'Reset Quota Counter' in the sidebar.")
            else:
                with st.chat_message("user", avatar="🧑‍💻"):
                    st.write(question_to_process)
                
                with st.chat_message("assistant", avatar="🧭"):
                    with st.spinner("Searching vector index and synthesizing file-grounded response..."):
                        result = answer_question(
                            st.session_state.vectorstore, question_to_process, google_api_key
                        )
                        increment_api_usage(1)
                    st.write(result["answer"])
                    
                    if result["sources"]:
                        with st.expander(f"📎 Grounded Citations ({len(result['sources'])} Retrieved Source Chunks)", expanded=True):
                            tab_labels = [f"📄 Chunk {s_idx+1}: {src['file_path']}" for s_idx, src in enumerate(result["sources"])]
                            source_tabs = st.tabs(tab_labels)
                            for s_idx, (src_tab, src) in enumerate(zip(source_tabs, result["sources"])):
                                with src_tab:
                                    loc = src["file_path"]
                                    lines_badge = f"Lines {src['start_line']} - {src['end_line']}" if src["start_line"] else "Entire File"
                                    lang = src.get("file_path", "").split(".")[-1]
                                    
                                    st.markdown(f"""
                                    <div style="margin-bottom: 12px; display: flex; align-items: center; flex-wrap: wrap;">
                                        <span class="badge-path">{loc}</span>
                                        <span class="badge-lines">{lines_badge}</span>
                                        <span class="badge-lang">{lang}</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    st.code(src["snippet"], language=lang if lang in ["py", "js", "ts", "java", "go", "cpp", "html", "md"] else None)

                st.session_state.chat_history.append({
                    "question": question_to_process,
                    "answer": result["answer"],
                    "sources": result["sources"]
                })


# ==============================================================================
# TAB 2: REPOSITORY STRUCTURE & AST SIGNATURE EXPLORER
# ==============================================================================
with tab_explorer:
    if not st.session_state.repo_path:
        st.info("👋 Index a repository first to explore its file hierarchy and detected function/class definitions.")
    else:
        st.markdown("### 🌳 Repository Structure & Signature Explorer")
        st.write("Browse the exact directory tree and top-level definitions extracted deterministically across your codebase.")
        
        with st.spinner("Extracting structural tree and AST signatures..."):
            summary_text = build_repo_summary(st.session_state.repo_path)
            
        # Parse out Directory Structure vs Key Definitions
        parts = summary_text.split("KEY DEFINITIONS PER FILE:")
        dir_tree = parts[0].replace("DIRECTORY STRUCTURE:\n", "").strip()
        definitions_text = parts[1].strip() if len(parts) > 1 else ""
        
        sub_tab_tree, sub_tab_sigs = st.tabs(["📁 Directory Hierarchy Tree", "⚡ Detected AST Signatures"])
        
        with sub_tab_tree:
            st.markdown("This tree represents all indexable files and directories discovered during indexing.")
            st.code(dir_tree, language=None)
            
        with sub_tab_sigs:
            st.markdown("Search and filter top-level function (`def`, `func`) and class definitions found across source files.")
            search_query = st.text_input("🔍 Filter by filename, class name, or function name...", placeholder="e.g., indexer, answer_question, class Chroma")
            
            if definitions_text:
                lines = [l.strip() for l in definitions_text.split("\n") if l.strip()]
                filtered_lines = [l for l in lines if not search_query or search_query.lower() in l.lower()]
                
                if not filtered_lines:
                    st.warning("No signatures matched your filter query.")
                else:
                    st.caption(f"Showing **{len(filtered_lines)}** file definition summaries:")
                    for l in filtered_lines:
                        # Format: "- file.py: def1, def2"
                        if ":" in l:
                            fname_part, sigs_part = l.split(":", 1)
                            fname_clean = fname_part.lstrip("- ").strip()
                            sigs_list = [s.strip() for s in sigs_part.split(",") if s.strip()]
                            
                            with st.expander(f"📄 **`{fname_clean}`** ({len(sigs_list)} symbols detected)", expanded=bool(search_query)):
                                st.markdown(f"**Path:** `{fname_clean}`")
                                st.write("**Detected Symbols:**")
                                st.code("\n".join([f"• {s}" for s in sigs_list]), language="py")
                        else:
                            st.write(l)
            else:
                st.info("No explicit function or class signatures were detected in this repository's language files.")


# ==============================================================================
# TAB 3: DOCUMENTATION & README STUDIO
# ==============================================================================
with tab_readme:
    if not st.session_state.repo_path:
        st.info("👋 Index a repository first to generate automated structural documentation.")
    else:
        st.markdown("### 📄 AI Documentation & README Studio")
        st.write("Generate a comprehensive, structurally accurate README section directly from your indexed codebase.")
        
        col_cfg1, col_cfg2 = st.columns([1, 1])
        with col_cfg1:
            doc_style = st.selectbox(
                "Documentation Tone & Target Style",
                [
                    "Technical & Comprehensive",
                    "Quickstart & Setup Guide",
                    "High-Level Architecture Overview",
                    "Beginner & Onboarding Guide"
                ]
            )
        with col_cfg2:
            custom_doc_instructions = st.text_input(
                "Additional Focus / Custom Instructions (Optional)",
                placeholder="e.g., Emphasize vector search, chroma persistence, and environment setup..."
            )

        if st.button("✍️ Generate Custom README Section", type="primary"):
            if not google_api_key:
                st.error("⚠️ Gemini API key not configured in environment (.env file).")
            elif not check_api_limit(1):
                used, limit = get_api_usage()
                st.error(f"⚠️ Daily API budget reached ({used}/{limit} requests used). Please click 'Reset Quota Counter' in the sidebar.")
            else:
                with st.spinner("Analyzing project structure, AST signatures, and synthesizing documentation..."):
                    readme_text = generate_readme_section(
                        st.session_state.repo_path,
                        google_api_key,
                        style=doc_style,
                        custom_instructions=custom_doc_instructions
                    )
                    increment_api_usage(1)
                    st.session_state.readme_output = readme_text
                st.success("README generated successfully!")

        if st.session_state.readme_output:
            st.divider()
            sub_readme_prev, sub_readme_raw = st.tabs(["🎨 Rendered Preview", "📝 Raw Markdown & Download"])
            
            with sub_readme_prev:
                st.markdown(st.session_state.readme_output)
                
            with sub_readme_raw:
                st.code(st.session_state.readme_output, language="markdown")
                st.download_button(
                    "⬇️ Download as README.md",
                    data=st.session_state.readme_output,
                    file_name="README_generated.md",
                    mime="text/markdown",
                    use_container_width=True
                )
