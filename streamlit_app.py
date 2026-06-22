import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="GPT Python Code Generator",
    page_icon="⚡",
    layout="wide"
)

st.title("GPT Python Code Generator")
st.markdown("An 85M parameter transformer built from scratch — type a Python prompt and watch it complete the code.")
st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    prompt = st.text_area(
        "Enter a Python prompt",
        value="def calculate_average(numbers):",
        height=100
    )

with col2:
    max_tokens = st.slider("Max tokens to generate", 50, 400, 200)
    temperature = st.slider("Temperature", 0.1, 1.5, 0.8,
                           help="Lower = more predictable, Higher = more creative")

if st.button("Generate code", use_container_width=True):
    with st.spinner("Generating..."):
        try:
            response = requests.post(f"{API_URL}/generate", json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            })
            result = response.json()
            st.divider()
            st.markdown("**Generated output:**")
            st.code(result["generated"], language="python")
            st.caption(f"Generated {result['tokens_generated']} tokens")
        except Exception as e:
            st.error(f"Error: {e} — make sure the API is running")

st.divider()
st.markdown("**How it works:**")
col1, col2, col3 = st.columns(3)
col1.metric("Parameters", "85M")
col2.metric("Training steps", "10,000")
col3.metric("Final train loss", "0.78")
st.caption("Built by Abhay Singh Wazir · Deakin University · github.com/Icarus-si")