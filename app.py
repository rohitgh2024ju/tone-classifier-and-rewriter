import streamlit as st
import requests
import html

# -------- CONFIG --------
API_URL = "https://politicly-winier-dona.ngrok-free.dev/predict"

st.set_page_config(page_title="Tone Analyzer", layout="centered")

# -------- SESSION STATE --------
if "result" not in st.session_state:
    st.session_state.result = None

# -------- GLOBAL JS (TOAST + COPY) --------
st.components.v1.html(
    """
<script>
function copyText(text) {
    navigator.clipboard.writeText(text);

    let old = document.getElementById("toast-msg");
    if (old) old.remove();

    const toast = document.createElement("div");
    toast.id = "toast-msg";
    toast.innerText = "Copied!";
    toast.style.position = "fixed";
    toast.style.bottom = "20px";
    toast.style.right = "20px";
    toast.style.background = "#22c55e";
    toast.style.color = "white";
    toast.style.padding = "10px 16px";
    toast.style.borderRadius = "10px";
    toast.style.fontSize = "14px";
    toast.style.boxShadow = "0 5px 20px rgba(0,0,0,0.3)";
    toast.style.zIndex = "9999";
    toast.style.opacity = "0";
    toast.style.transform = "translateY(10px)";
    toast.style.transition = "all 0.3s ease";

    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = "1";
        toast.style.transform = "translateY(0)";
    }, 50);

    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transform = "translateY(10px)";
        setTimeout(() => toast.remove(), 300);
    }, 2000);
}
</script>
""",
    height=0,
)

# -------- CSS --------
st.markdown(
    """
<style>
body {
    background-color: #0f172a;
}

.block-container {
    padding-top: 2rem;
    max-width: 720px;
}

h1 {
    text-align: center;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 2rem;
}

.stButton > button {
    border-radius: 10px;
    height: 45px;
    font-weight: 500;
}

.card-row {
    display:flex;
    align-items:center;
    gap:10px;
    margin-bottom:12px;
}

.card {
    flex:1;
    padding:14px 16px;
    border-radius:14px;
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(255,255,255,0.05);
    font-size:15px;
    transition: all 0.2s ease;
}

.card:hover {
    transform: scale(1.01);
    border: 1px solid rgba(255,255,255,0.1);
}

.copy-btn {
    height:44px;
    width:44px;
    border-radius:10px;
    border:1px solid rgba(255,255,255,0.1);
    background:#1e293b;
    cursor:pointer;
    transition: all 0.2s ease;
}

.copy-btn:hover {
    background:#334155;
}
</style>
""",
    unsafe_allow_html=True,
)

# -------- HEADER --------
st.markdown("<h1>🧠 Tone Analyzer</h1>", unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Rewrite your text smarter, not harder</div>',
    unsafe_allow_html=True,
)

# -------- INPUT --------
text = st.text_area("✍️ Enter your text", height=150)

target_tone = st.selectbox(
    "🎯 Target Tone", ["Professional", "Friendly", "Polite", "Formal", "Assertive"]
)


# -------- API --------
def call_api(payload):
    try:
        return requests.post(API_URL, json=payload, timeout=60)
    except:
        return None


# -------- ANALYZE --------
if st.button("🚀 Analyze", use_container_width=True):

    if not text.strip():
        st.warning("Please enter text")
    else:
        with st.spinner("Analyzing..."):
            response = call_api({"text": text, "target_tone": target_tone})

            if response and response.status_code == 200:
                st.session_state.result = response.json()
            else:
                st.error("❌ API error or server unreachable")

# -------- CLEAR --------
if st.session_state.result:
    if st.button("🗑️ Clear Results"):
        st.session_state.result = None
        st.rerun()

# -------- RESULTS --------
if st.session_state.result:

    data = st.session_state.result

    st.divider()

    # Tone
    st.subheader("📊 Detected Tone")
    st.success(data["detected_tone"])

    # Confidence
    st.subheader("🎯 Confidence")
    for pred in data["predictions"]:
        st.write(f"{pred['label']} — {pred['confidence']}%")
        st.progress(pred["confidence"] / 100)

    # Suggestions
    st.subheader("✍️ Suggestions")

    for i, suggestion in enumerate(data["suggestions"]):

        safe_text = html.escape(suggestion)

        col1, col2 = st.columns([8, 1])

        with col1:
            st.markdown(f"""
            <div style="
                padding:14px 16px;
                border-radius:14px;
                background: rgba(30, 41, 59, 0.7);
                border: 1px solid rgba(255,255,255,0.05);
                font-size:15px;
            ">
                {suggestion}
            </div>
            """, unsafe_allow_html=True)

        with col2:
            if st.button("📋", key=f"copy_{i}"):

                # 🔥 This is the ONLY reliable way
                st.write("")  # trigger render safely

                st.components.v1.html(f"""
                    <script>
                    navigator.clipboard.writeText("{safe_text}");
                    </script>
                """, height=0)

                st.toast("Copied!", icon="✅")
