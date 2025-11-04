import os
import time

import requests
import streamlit as st

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

st.set_page_config(page_title="SMS Spam Classifier", page_icon="📨", layout="centered")

st.title("📨 SMS Spam Classifier")
st.caption("Kaggle-trained • CPU-only • FastAPI + Streamlit")

with st.sidebar:
    st.subheader("Settings")
    st.text_input(
        "FASTAPI_URL",
        value=FASTAPI_URL,
        key="api_url",
        help="Backend inference endpoint",
    )
    st.markdown("---")
    st.write("Examples")
    ex1 = "Congratulations! You have won a free prize. Claim now."
    ex2 = "Are we still meeting at 5 pm today?"
    ex3 = "URGENT! Call now to verify your account and win $1000."
    if st.button("Use Example 1"):
        st.session_state["text"] = ex1
    if st.button("Use Example 2"):
        st.session_state["text"] = ex2
    if st.button("Use Example 3"):
        st.session_state["text"] = ex3

st.text_area("Input text", key="text", height=150, placeholder="Type an SMS message...")

col1, col2 = st.columns([1, 3])
with col1:
    go = st.button("Predict", type="primary")
with col2:
    st.write("")

if go:
    txt = st.session_state.get("text", "")
    if not txt.strip():
        st.error("Please enter text.")
    else:
        start = time.time()
        try:
            response = requests.post(
                f"{st.session_state['api_url'].rstrip('/')}/predict",
                json={"text": txt},
                timeout=10,
            )
            latency = int((time.time() - start) * 1000)
            if response.status_code != 200:
                st.error(f"Error {response.status_code}: {response.text}")
            else:
                data = response.json()
                label = data.get("prediction")
                score = data.get("score")
                model_version = data.get("model_version")
                st.success(f"Prediction: **{label}**  •  Score: **{score:.3f}**")
                st.caption(
                    f"Latency: {data.get('latency_ms', latency)} ms  •  Model: {model_version}"
                )

                with st.expander("Why? Top contributing tokens"):
                    features = data.get("debug", {}).get("top_features", [])
                    if features:
                        st.table({
                            "token": [token for token, _ in features],
                            "contrib": [value for _, value in features],
                        })
                    else:
                        st.write("No contributing tokens found.")
        except Exception as exc:
            st.error(f"Request failed: {exc}")

st.markdown("---")
st.caption("Powered by FastAPI + Streamlit • Artifacts from Kaggle training")
