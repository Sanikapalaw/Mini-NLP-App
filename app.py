import streamlit as st
from transformers import pipeline

# Config
st.set_page_config(page_title="NLP ToolKit (Fast)", layout="wide")
st.title("NLP ToolKit — Fast Model Edition")
st.caption("Powered by `google/flan-t5-small` for faster CPU inference.")

# The single model used for everything
# Switched to 'small' to reduce runtime latency on CPUs
MODEL_CHECKPOINT = "google/flan-t5-small"

# Mapping tasks to the prefixes FLAN-T5 expects
TASK_PROMPTS = {
    "Text generation": "", # No prefix, just generates text
    "Summarization": "summarize: ",
    "Sentiment analysis": "classify sentiment: ",
    "Named Entity Recognition (NER)": "find entities: ",
    "Question Answering (QA)": "answer: ", # We will format this specifically in code
    "Translation (en→fr)": "translate English to French: ",
    "Paraphrase": "paraphrase: ",
    "Grammar correction": "fix grammar: ",
}

@st.cache_resource
def get_pipeline():
    """Load the single universal model."""
    return pipeline("text2text-generation", model=MODEL_CHECKPOINT)

left, right = st.columns(2)

with left:
    st.header("Input & Options")
    task = st.selectbox("Choose task", list(TASK_PROMPTS.keys()), index=0)

    # Dynamic UI: Change label and placeholder based on task
    if task == "Text generation":
        lbl = "Enter a Prompt"
        ph = "e.g., Once upon a time, there was a robot who..."
    elif task == "Question Answering (QA)":
        lbl = "Context Text"
        ph = "Paste the story or article here..."
    else:
        lbl = "Input Text"
        ph = "Paste text here..."

    input_text = st.text_area(lbl, height=220, placeholder=ph)
    
    question = ""
    if task == "Question Answering (QA)":
        question = st.text_input("Question", placeholder="What are you looking for in the text?")

    # Options for generation
    max_new_tokens = st.slider("Max generation length", 10, 300, 100, 10)
    
    run = st.button("Run")

with right:
    st.header("Output")
    out = st.empty()

if run:
    if not input_text:
        out.warning(f"⚠️ Please provide text for '{task}'.")
    else:
        with st.spinner(f"Running {MODEL_CHECKPOINT}..."):
            try:
                pipe = get_pipeline()
                
                # Construct the prompt based on the task
                prefix = TASK_PROMPTS[task]
                
                if task == "Question Answering (QA)":
                    if not question:
                        out.error("Please enter a question.")
                        st.stop()
                    # FLAN-T5 QA format
                    final_input = f"question: {question} context: {input_text}"
                else:
                    final_input = f"{prefix}{input_text}"

                # Run the single model
                res = pipe(final_input, max_new_tokens=max_new_tokens)
                
                # Display result
                generated_text = res[0]['generated_text']
                out.success("Success")
                out.markdown(f"**Result:**")
                out.write(generated_text)
                
                # Debug info to show user what actually happened
                with st.expander("See raw input sent to model"):
                    st.code(final_input)

            except Exception as e:
                out.error(f"Error: {str(e)}")
