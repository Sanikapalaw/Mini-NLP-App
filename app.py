import streamlit as st
from transformers import pipeline

# Config
st.set_page_config(page_title="NLP ToolKit (One Model)", layout="wide")
st.title("NLP ToolKit — Single Model Edition")
st.caption("Powered by `google/flan-t5-base` for all tasks.")

# The single model used for everything
MODEL_CHECKPOINT = "google/flan-t5-base"

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
    input_text = st.text_area("Input text / Context", height=220, placeholder="Paste text here...")
    
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
        out.error("Please provide input text.")
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
