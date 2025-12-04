# app.py
"""
NLP ToolKit — Compact multi-task Streamlit app using Hugging Face pipelines.
Supports: text generation, summarization, sentiment, NER, QA, translation (en->fr),
paraphrase, grammar correction, and (optional) sentence similarity.
Save as app.py and run: streamlit run app.py
"""

import streamlit as st
from functools import lru_cache

# Transformers
from transformers import pipeline

# Optional: sentence-transformers for similarity
try:
    from sentence_transformers import SentenceTransformer, util
    SB_AVAILABLE = True
except Exception:
    SB_AVAILABLE = False

st.set_page_config(title="NLP ToolKit", layout="wide")
st.title("NLP ToolKit")
st.caption("Multi-task NLP playground — left: input/options, right: output (side-by-side)")

# Mapping of user-visible tasks to pipeline tasks and default models
MODEL_MAP = {
    "Text generation": ("text-generation", "gpt2"),
    "Summarization": ("summarization", "facebook/bart-large-cnn"),
    "Sentiment analysis": ("sentiment-analysis", None),
    "Named Entity Recognition (NER)": ("ner", "dslim/bert-base-NER"),
    "Question Answering (QA)": ("question-answering", "distilbert-base-cased-distilled-squad"),
    "Translation (en→fr)": ("translation", "Helsinki-NLP/opus-mt-en-fr"),
    "Paraphrase (T5)": ("text2text-generation", "t5-small"),
    "Grammar correction": ("text2text-generation", "prithivida/grammar_error_correcter_v1"),
    "Similarity (SBERT)": ("similarity", None),
}

@st.cache_resource
def get_pipeline(task, model_name=None, **kwargs):
    """Return cached transformers pipeline. model_name may be None to use defaults."""
    if model_name:
        return pipeline(task, model=model_name, **kwargs)
    return pipeline(task, **kwargs)

# Two columns side-by-side
left, right = st.columns(2)

with left:
    st.header("Input & Options")
    task = st.selectbox("Choose task", list(MODEL_MAP.keys()), index=0)
    input_text = st.text_area("Input text / Context", height=220, placeholder="Paste text, prompt or context here...")
    # Task-specific inputs
    question = ""
    second_sentence = ""
    if task == "Question Answering (QA)":
        question = st.text_input("Question (for QA)", placeholder="e.g. Where is the Taj Mahal?")
    if task == "Text generation":
        max_length = st.slider("max_length", min_value=50, max_value=600, value=150, step=10)
        do_sample = st.checkbox("do_sample", value=False)
        num_return_sequences = st.number_input("num_return_sequences", min_value=1, max_value=3, value=1, step=1)
    if task == "Summarization":
        min_len = st.number_input("min_length", min_value=5, max_value=200, value=10, step=1)
        max_len = st.number_input("max_length", min_value=10, max_value=600, value=80, step=1)
    if task == "Similarity (SBERT)":
        if not SB_AVAILABLE:
            st.warning("Install sentence-transformers to enable similarity (`pip install sentence-transformers`).")
        second_sentence = st.text_input("Sentence B (to compare)", placeholder="Enter sentence to compare to")
    run = st.button("Run")

with right:
    st.header("Output")
    out = st.empty()

if run:
    # Basic validation
    if task != "Text generation" and not input_text:
        out.error("Please provide input text in the left column.")
    else:
        try:
            task_type, model_name = MODEL_MAP[task]

            # Similarity branch (uses sentence-transformers)
            if task == "Similarity (SBERT)":
                if not SB_AVAILABLE:
                    out.error("sentence-transformers not available. Install with `pip install sentence-transformers`.")
                elif not second_sentence:
                    out.info("Enter the second sentence to compute similarity.")
                else:
                    sbert = SentenceTransformer("all-MiniLM-L6-v2")
                    a = sbert.encode(input_text, convert_to_tensor=True)
                    b = sbert.encode(second_sentence, convert_to_tensor=True)
                    score = util.pytorch_cos_sim(a, b).item()
                    out.markdown(f"**Cosine similarity:** `{score:.4f}`")
                    out.write("**Sentence A:**")
                    out.write(input_text)
                    out.write("**Sentence B:**")
                    out.write(second_sentence)

            elif task == "Named Entity Recognition (NER)":
                pipe = get_pipeline("ner", model_name, aggregation_strategy="simple")
                entities = pipe(input_text)
                if not entities:
                    out.info("No entities found.")
                else:
                    for e in entities:
                        word = e.get("word") or e.get("entity")
                        grp = e.get("entity_group") or e.get("entity")
                        score = e.get("score", 0.0)
                        out.markdown(f"- **{word}** — {grp} (score: {score:.3f})")

            elif task == "Question Answering (QA)":
                if not question:
                    out.error("Please provide a question for QA.")
                else:
                    pipe = get_pipeline("question-answering", model_name)
                    res = pipe(question=question, context=input_text)
                    out.markdown("**Answer:**")
                    out.write(res.get("answer"))
                    out.write(res)

            elif task == "Text generation":
                pipe = get_pipeline("text-generation", model_name)
                gen = pipe(input_text or "", max_length=max_length, do_sample=do_sample, num_return_sequences=num_return_sequences)
                for i, g in enumerate(gen, start=1):
                    out.markdown(f"**Generated #{i}:**")
                    out.code(g.get("generated_text", ""))

            elif task == "Summarization":
                pipe = get_pipeline("summarization", model_name)
                summ = pipe(input_text, max_length=max_len, min_length=min_len, do_sample=False)
                out.markdown("**Summary**")
                out.write(summ[0].get("summary_text", str(summ)))

            elif task == "Sentiment analysis":
                pipe = get_pipeline("sentiment-analysis", model_name)
                res = pipe(input_text)
                out.write(res)

            elif task == "Translation (en→fr)":
                pipe = get_pipeline("translation", model_name)
                res = pipe(input_text)
                # translation pipeline returns list of dicts with 'translation_text'
                if isinstance(res, list) and "translation_text" in res[0]:
                    out.write(res[0]["translation_text"])
                else:
                    out.write(res)

            elif task == "Paraphrase (T5)":
                pipe = get_pipeline("text2text-generation", model_name)
                prompt = f"paraphrase: {input_text}"
                res = pipe(prompt, max_length=256, num_return_sequences=1)
                out.write(res[0]["generated_text"])

            elif task == "Grammar correction":
                pipe = get_pipeline("text2text-generation", model_name)
                res = pipe(input_text, max_length=256)
                out.write(res[0]["generated_text"])

            else:
                out.info("Task not implemented.")

        except Exception as e:
            out.error(f"Error: {e}")
            st.exception(e)

st.markdown("---")
st.caption("Notes: Models download on first run (internet required). If memory is limited, avoid running large models locally.")
