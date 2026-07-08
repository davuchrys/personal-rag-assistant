import os
import json
import uuid
import hashlib
import bcrypt
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

from src.rag_pipeline import RAGPipeline

def format_confidence(distance: float):
    """Return a confidence label and hex color based on distance.
    Lower distance = higher confidence.
    """
    if distance <= 0.45:
        return "High confidence", "#4caf50"
    if distance <= 0.75:
        return "Medium confidence", "#ff9800"
    if distance <= 1.0:
        return "Low confidence", "#f44336"
    return "Insufficient evidence", "#9e9e9e"


# --- Setup ---
st.set_page_config(page_title="RAG Assistant", page_icon="📎", layout="wide")

SESSIONS_FILE = "data/sessions.json"
os.makedirs("data", exist_ok=True)

# GUARDRAIL: reject uploads above this size to avoid excessive memory use
# during PDF parsing / embedding, and to cap per-user storage growth.
MAX_UPLOAD_SIZE_MB = 20

# ---- Supabase Auth ----
@st.cache_resource
def _get_supabase():
    """Initialize Supabase client from Streamlit Secrets or env vars."""
    from supabase import create_client
    try:
        url = st.secrets.get("SUPABASE_URL") or os.getenv("SUPABASE_URL", "")
        key = st.secrets.get("SUPABASE_KEY") or os.getenv("SUPABASE_KEY", "")
    except Exception:
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_KEY", "")
    if not url or not key:
        return None
    return create_client(url, key)

def _hash_password(password: str) -> str:
    """Bcrypt is salted per-password and deliberately slow, unlike plain SHA256."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def _verify_password(password: str, stored_hash: str) -> bool:
    """Verifies a password against a stored hash.

    Supports legacy unsalted SHA256 hashes (from before the bcrypt migration)
    so existing accounts keep working — a successful legacy match triggers a
    transparent re-hash to bcrypt in the login flow below.
    """
    if stored_hash.startswith(("$2a$", "$2b$", "$2y$")):
        try:
            return bcrypt.checkpw(password.encode(), stored_hash.encode())
        except ValueError:
            return False
    return hashlib.sha256(password.encode()).hexdigest() == stored_hash

def _is_legacy_hash(stored_hash: str) -> bool:
    return not stored_hash.startswith(("$2a$", "$2b$", "$2y$"))

def _load_users() -> dict:
    """Load all users from Supabase. Falls back to empty dict if unavailable."""
    sb = _get_supabase()
    if sb is None:
        return {}
    try:
        result = sb.table("users").select("username, password_hash").execute()
        return {row["username"]: row["password_hash"] for row in result.data}
    except Exception:
        return {}

def _save_user(username: str, password_hash: str) -> bool:
    """Save a new user to Supabase. Returns True on success."""
    sb = _get_supabase()
    if sb is None:
        return False
    try:
        sb.table("users").insert({"username": username, "password_hash": password_hash}).execute()
        return True
    except Exception:
        return False

def _update_user_password(username: str, password_hash: str) -> None:
    """Updates a user's stored hash. Used to migrate legacy SHA256 hashes to bcrypt on login."""
    sb = _get_supabase()
    if sb is None:
        return
    try:
        sb.table("users").update({"password_hash": password_hash}).eq("username", username).execute()
    except Exception:
        pass

def _create_session(username: str) -> str:
    """Create a session token and persist it to Supabase."""
    token = str(uuid.uuid4())
    sb = _get_supabase()
    if sb:
        try:
            sb.table("sessions").insert({"token": token, "username": username}).execute()
        except Exception:
            pass
    return token

def _get_session_user(token: str) -> str | None:
    """Look up username from a session token in Supabase."""
    sb = _get_supabase()
    if sb is None or not token:
        return None
    try:
        result = sb.table("sessions").select("username").eq("token", token).execute()
        if result.data:
            return result.data[0]["username"]
    except Exception:
        pass
    return None

def _delete_session(token: str):
    """Remove a session token from Supabase on logout."""
    sb = _get_supabase()
    if sb and token:
        try:
            sb.table("sessions").delete().eq("token", token).execute()
        except Exception:
            pass

def _login_user(username: str):
    """Log in a user: set session state and persist token in URL."""
    token = _create_session(username)
    st.session_state.username = username
    st.query_params["token"] = token

# Auto-login from URL token on refresh
if "username" not in st.session_state or not st.session_state.username:
    token = st.query_params.get("token")
    if token:
        user = _get_session_user(token)
        if user:
            st.session_state.username = user


if "username" not in st.session_state or not st.session_state.username:
    st.markdown("<h2 style='text-align: center; margin-top: 5rem;'>Welcome to RAG Assistant</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Sign up or log in to access your private workspace.</p>", unsafe_allow_html=True)

    if _get_supabase() is None:
        st.warning("⚠️ Database not configured. Set SUPABASE_URL and SUPABASE_KEY in Secrets / .env to enable accounts.")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

        with tab_login:
            with st.form("login_form"):
                login_user = st.text_input("Username", key="login_user")
                login_pass = st.text_input("Password", type="password", key="login_pass")
                login_submit = st.form_submit_button("Login", use_container_width=True, type="primary")
                if login_submit:
                    u = login_user.strip()
                    if not u or not login_pass:
                        st.error("Please enter both username and password.")
                    else:
                        users = _load_users()
                        if u not in users or not _verify_password(login_pass, users[u]):
                            st.error("Invalid username or password.")
                        else:
                            # Transparently migrate legacy SHA256 hashes to bcrypt on successful login
                            if _is_legacy_hash(users[u]):
                                _update_user_password(u, _hash_password(login_pass))
                            _login_user(u)
                            st.rerun()

        with tab_signup:
            with st.form("signup_form"):
                signup_user = st.text_input("Username", key="signup_user")
                signup_pass = st.text_input("Password", type="password", key="signup_pass")
                signup_pass2 = st.text_input("Confirm Password", type="password", key="signup_pass2")
                signup_submit = st.form_submit_button("Create Account", use_container_width=True, type="primary")
                if signup_submit:
                    u = signup_user.strip()
                    if not u or not signup_pass:
                        st.error("Please fill in all fields.")
                    elif len(signup_pass) < 4:
                        st.error("Password must be at least 4 characters.")
                    elif signup_pass != signup_pass2:
                        st.error("Passwords do not match.")
                    else:
                        users = _load_users()
                        if u in users:
                            st.error("Username already taken. Please choose another.")
                        elif not _save_user(u, _hash_password(signup_pass)):
                            st.error("Could not save account. Please check Supabase configuration.")
                        else:
                            _login_user(u)
                            st.rerun()
    st.stop()

USERNAME = st.session_state.username

@st.cache_resource
def get_pipeline(username: str):
    return RAGPipeline(vector_db_path=f"./vector_db/{username}")

pipeline = get_pipeline(USERNAME)
CHATS_DIR = f"data/chats/{USERNAME}"
os.makedirs(CHATS_DIR, exist_ok=True)

def save_chat(session_id, messages, summary_state=None):
    if not messages:
        return
    file_path = os.path.join(CHATS_DIR, f"{session_id}.json")
    payload = {"messages": messages, "summary": summary_state or {"text": "", "covered": 0}}
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def load_chat(session_id):
    """Returns (messages, summary_state). Handles old files that were a plain list."""
    file_path = os.path.join(CHATS_DIR, f"{session_id}.json")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data, {"text": "", "covered": 0}
        return data.get("messages", []), data.get("summary", {"text": "", "covered": 0})
    return [], {"text": "", "covered": 0}

def get_all_chats():
    chats = []
    for filename in os.listdir(CHATS_DIR):
        if filename.endswith(".json"):
            session_id = filename[:-5]
            file_path = os.path.join(CHATS_DIR, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    messages = json.load(f)
                    if messages:
                        first_msg = next((m["content"] for m in messages if m["role"] == "user"), "Empty Chat")
                        title = first_msg[:20] + "..." if len(first_msg) > 20 else first_msg
                        mtime = os.path.getmtime(file_path)
                        chats.append({"session_id": session_id, "title": title, "mtime": mtime})
                except Exception:
                    pass
    chats.sort(key=lambda x: x["mtime"], reverse=True)
    return chats

# Initialize chat history in session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages, st.session_state.summary_state = load_chat(st.session_state.session_id)

# --- Clean, Minimal CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 18px !important;
    }
    
    .stApp {
        background-color: #fafafa;
    }
    
    /* Clean header */
    .app-header {
        padding: 2rem 0 1.5rem 0;
        border-bottom: 1px solid #eee;
        margin-bottom: 1.5rem;
    }
    .app-header h1 {
        font-size: 2.2rem;
        font-weight: 600;
        color: #111;
        margin: 0 0 0.25rem 0;
    }
    .app-header p {
        font-size: 1.2rem;
        color: #666;
        margin: 0;
    }
    
    /* Status bar */
    .status-bar {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0.8rem;
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        font-size: 1.0rem;
        color: #15803d;
        margin-top: 0.75rem;
    }
    .status-bar.empty {
        background: #fefce8;
        border-color: #fef08a;
        color: #a16207;
    }
    
    /* Source tag */
    .source-tag {
        display: inline-block;
        background: #f1f5f9;
        color: #475569;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.95rem;
        font-weight: 500;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
        border: 1px solid #e2e8f0;
    }
    
    /* Chat styling */
    .chat-answer {
        line-height: 1.7;
        color: #1a1a1a;
        font-size: 1.1rem;
    }
    
    /* Sidebar clean */
    [data-testid="stSidebar"] {
        background-color: #fff;
        border-right: 1px solid #eee;
    }
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] button {
        font-size: 15px !important;
    }
    
    .sidebar-section {
        font-size: 12px !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #999 !important;
        font-weight: 600 !important;
        margin-bottom: 0.25rem;
    }
    
    .file-item {
        padding: 0.25rem 0.5rem;
        background: #f8f8f8;
        border-radius: 6px;
        font-size: 14px !important;
        color: #333 !important;
        margin-bottom: 0.25rem;
    }
    
    /* Nicer buttons */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.15s ease !important;
    }
    
    [data-testid="stSidebar"] .stButton > button {
        padding-top: 0.35rem !important;
        padding-bottom: 0.35rem !important;
        min-height: 2.2rem !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px) !important;
    }
    
    /* Chunk detail */
    .chunk-meta {
        font-size: 0.95rem;
        color: #888;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown(f"**Logged in as: `{USERNAME}`**")
    if st.button("🚪 Logout", use_container_width=True, type="secondary"):
        # Remove session token from Supabase
        token = st.query_params.get("token")
        if token:
            _delete_session(token)
        st.query_params.clear()
        st.session_state.username = None
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.summary_state = {"text": "", "covered": 0}
        st.rerun()

    st.divider()

    # New Chat Button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.summary_state = {"text": "", "covered": 0}
        st.rerun()

    st.markdown('<div class="sidebar-section" style="margin-top:1rem;">Past Chats</div>', unsafe_allow_html=True)
    past_chats = get_all_chats()
    for chat in past_chats:
        is_active = (chat["session_id"] == st.session_state.get("session_id"))
        btn_type = "primary" if is_active else "secondary"

        col1, col2 = st.columns([0.85, 0.15], gap="small")
        with col1:
            if st.button(chat['title'], key=f"chat_{chat['session_id']}", use_container_width=True, type=btn_type):
                st.session_state.session_id = chat["session_id"]
                st.session_state.messages, st.session_state.summary_state = load_chat(chat["session_id"])
                st.rerun()
        with col2:
            if st.button("✕", key=f"del_{chat['session_id']}", use_container_width=True, type="tertiary"):
                file_path = os.path.join(CHATS_DIR, f"{chat['session_id']}.json")
                if os.path.exists(file_path):
                    os.remove(file_path)
                if is_active:
                    st.session_state.session_id = str(uuid.uuid4())
                    st.session_state.messages = []
                    st.session_state.summary_state = {"text": "", "covered": 0}
                st.rerun()

    st.divider()
    
    st.markdown('<div class="sidebar-section">📁 Document Workspace</div>', unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Add files", 
        type=['pdf', 'txt', 'md'], 
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        for f in uploaded_files:
            st.markdown(f'<div class="file-item">📄 {f.name}</div>', unsafe_allow_html=True)

        if st.button("Index Documents", use_container_width=True):
            file_paths = []
            oversized = []
            for file in uploaded_files:
                # GUARDRAIL: reject oversized uploads before writing them to disk
                # or feeding them into the (memory-hungry) PDF/embedding pipeline.
                if file.size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                    oversized.append(file.name)
                    continue
                file_path = os.path.join("data", file.name)
                with open(file_path, "wb") as f_out:
                    f_out.write(file.getbuffer())
                file_paths.append(file_path)

            if oversized:
                st.error(f"Skipped (over {MAX_UPLOAD_SIZE_MB}MB limit): {', '.join(oversized)}")

            if file_paths:
                with st.spinner("Indexing..."):
                    num_chunks = pipeline.ingest_files(file_paths)
                    st.success(f"Done — {num_chunks} chunks indexed.")

    # Show indexed documents and management options
    indexed_files_db = pipeline.get_indexed_files()
    if indexed_files_db:
        st.markdown('<div class="sidebar-section" style="margin-top: 1rem;">Indexed Files</div>', unsafe_allow_html=True)
        for fname in indexed_files_db:
            st.markdown(f'<div class="file-item" style="border-left: 3px solid #4caf50;">✓ {fname}</div>', unsafe_allow_html=True)
            
        if st.button("🗑️ Clear Indexed Data", use_container_width=True):
            pipeline.clear_index()
            st.success("✅ Index cleared. You can now upload new documents.")
            st.session_state.messages = []
            st.session_state.summary_state = {"text": "", "covered": 0}
            st.rerun()
    
    st.divider()
    
    with st.expander("⚙️ Advanced Settings"):
        st.markdown('<div class="sidebar-section" style="margin-top:0.5rem;">Search Strictness</div>', unsafe_allow_html=True)
        distance_threshold = st.slider(
            "Threshold", 
            min_value=0.1, max_value=2.0, value=0.7, step=0.1,
            help="Lower = stricter matching. Higher = more lenient."
        )
        debug_mode = st.toggle("Show retrieved chunks", help="Displays the exact text chunks used to answer.")

        if debug_mode:
            summary_state = st.session_state.get("summary_state", {"text": "", "covered": 0})
            st.markdown('<div class="sidebar-section" style="margin-top:1rem;">Long-term Memory Summary</div>', unsafe_allow_html=True)
            st.caption(f"Messages folded into summary: {summary_state.get('covered', 0)} / {len(st.session_state.get('messages', []))}")
            if summary_state.get("text"):
                st.info(summary_state["text"])
            else:
                st.caption("No summary yet — appears once this chat passes 20 messages.")

# --- Main Content ---

# Header
db_count = pipeline.get_document_count()

indexed_files = pipeline.get_indexed_files()
files_text = ""
if indexed_files:
    files_text = f"<div style='margin-top: 0.75rem; font-size: 1.1rem; color: #64748b;'><strong>📄 Uploaded Files:</strong> {', '.join(indexed_files)}</div>"

st.markdown(f"""
<div class="app-header">
    <h1>📎 RAG Assistant</h1>
    <p>Ask questions — answers come only from your documents.</p>
    <div class="status-bar {'empty' if db_count == 0 else ''}">
        {'⚠️ No documents indexed yet' if db_count == 0 else f'✓ {db_count} chunks ready'}
    </div>
    {files_text}
</div>
""", unsafe_allow_html=True)


# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.write(msg["content"])
        else:
            # Assistant message with answer + sources
            st.markdown(f'<div class="chat-answer">{msg["answer"]}</div>', unsafe_allow_html=True)
            
            if msg.get("sources"):
                st.markdown("---")
                sources_html = " ".join([f'<span class="source-tag">📄 {s}</span>' for s in msg["sources"]])
                st.markdown(f"**Sources:** {sources_html}", unsafe_allow_html=True)
            
            # Debug chunks inside the message
            if msg.get("chunks") and debug_mode:
                with st.expander("Retrieved chunks"):
                    for i, chunk in enumerate(msg["chunks"]):
                        filename = chunk['metadata'].get('filename', 'Unknown')
                        dist = chunk['distance']
                        st.markdown(f'<div class="chunk-meta">Chunk {i+1} · {filename} · distance: {dist:.4f}</div>', unsafe_allow_html=True)
                        st.text(chunk['text'][:500])
                        if i < len(msg["chunks"]) - 1:
                            st.divider()

# Chat input
if query := st.chat_input("Ask something from your documents..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    save_chat(st.session_state.session_id, st.session_state.messages, st.session_state.summary_state)
    with st.chat_message("user"):
        st.write(query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Reading your documents..."):
            result = pipeline.ask(
                query=query,
                distance_threshold=distance_threshold,
                chat_history=st.session_state.messages,
                summary=st.session_state.summary_state.get("text"),
            )
            answer = result["answer"]
            chunks = result["context_chunks"]
            
            fallback = "I could not find enough information in the uploaded documents."
            
            if fallback in answer:
                st.warning(answer)
                sources = []
            else:
                st.markdown(f'<div class="chat-answer">{answer}</div>', unsafe_allow_html=True)
                sources = list(set(c['metadata'].get('filename', 'Unknown') for c in chunks))
                
                if sources:
                    st.markdown("---")
                    sources_html = " ".join([f'<span class="source-tag">📄 {s}</span>' for s in sources])
                    st.markdown(f"**Sources:** {sources_html}", unsafe_allow_html=True)
            
            if chunks and debug_mode:
                with st.expander("Retrieved chunks"):
                    for i, chunk in enumerate(chunks):
                        filename = chunk['metadata'].get('filename', 'Unknown')
                        dist = chunk['distance']
                        st.markdown(f'<div class="chunk-meta">Chunk {i+1} · {filename} · distance: {dist:.4f}</div>', unsafe_allow_html=True)
                        st.text(chunk['text'][:500])
                        if i < len(chunks) - 1:
                            st.divider()
            
            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "answer": answer,
                "sources": sources,
                "chunks": chunks
            })
            # Refresh the rolling long-term summary once history grows long enough
            st.session_state.summary_state = pipeline.maybe_update_summary(
                st.session_state.messages, st.session_state.summary_state
            )
            save_chat(st.session_state.session_id, st.session_state.messages, st.session_state.summary_state)
