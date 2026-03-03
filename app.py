import streamlit as st
import requests

# -------- CONFIG --------
API_URL = "https://politicly-winier-dona.ngrok-free.dev/predict"

st.set_page_config(page_title="Tone Analyzer", layout="centered")

# -------- SESSION STATE --------
if "result" not in st.session_state:
    st.session_state.result = None

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
        with st.spinner("Analyzing... (first request may be slow)"):

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

# -------- EMPTY STATE --------
if not st.session_state.result:
    st.info("Enter text and click Analyze to see results")

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

        col1, col2 = st.columns([6, 1])

        with col1:
            st.markdown(
                f"""
            <div style="
                padding:14px 16px;
                border-radius:14px;
                background: rgba(30, 41, 59, 0.7);
                border: 1px solid rgba(255,255,255,0.05);
                font-size:15px;
                margin-bottom:10px;
            ">
                {suggestion}
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            if st.button("📋", key=f"copy_{i}"):
                st.toast("Copied! Select below and press Ctrl+C", icon="✅")

                st.text_area(
                    "Copy text:", value=suggestion, height=120, key=f"copy_area_{i}"
                )
