import streamlit as st
from transformers import pipeline

# Optional: sentence-transformers
try:
    from sentence_transformers import SentenceTransformer, util
    SB_AVAILABLE = True
except ImportError:
    SB_AVAILABLE = False

st.set_page_config(title="NLP ToolKit", layout="wide")
st.title("NLP ToolKit")

MODEL_MAP = {
    "Text generation": ("text-generation", "gpt2"),
    "Summarization": ("summarization", "facebook/bart-large-cnn"),
    "Sentiment analysis": ("sentiment-analysis", "distilbert-base-uncased-finetuned-sst-2-english"),
    "Named Entity Recognition (NER)": ("ner", "dslim/bert-base-NER"),
    "Question Answering (QA)": ("question-answering", "distilbert-base-cased-distilled-squad"),
    "Translation (en→fr)": ("translation", "Helsinki-NLP/opus-mt-en-fr"),
    "Paraphrase (T5)": ("text2text-generation", "t5-small"),
    "Grammar correction": ("text2text-generation", "prithivida/grammar_error_correcter_v1"),
    "Similarity (SBERT)": ("similarity", None),
}

@st.cache_resource
def get_pipeline(task, model_name=None, **kwargs):
    if model_name:
        return pipeline(task, model=model_name, **kwargs)
    return pipeline(task, **kwargs)

@st.cache_resource
def get_sbert_model():
    if SB_AVAILABLE:
        return SentenceTransformer("all-MiniLM-L6-v2")
    return None

left, right = st.columns(2)

with left:
    st.header("Input & Options")
    task = st.selectbox("Choose task", list(MODEL_MAP.keys()), index=0)
    input_text = st.text_area("Input text", height=220, placeholder="Paste text here...")
    
    question = ""
    second_sentence = ""
    
    if task == "Question Answering (QA)":
        question = st.text_input("Question", placeholder="Ask a question about the context")
    
    if task == "Text generation":
        # Renamed to max_new_tokens for clarity and crash prevention
        max_new_tokens = st.slider("New tokens to generate", 10, 200, 50, 10)
        do_sample = st.checkbox("do_sample", value=True) # Usually True is better for generation
        
    if task == "Summarization":
        min_len = st.number_input("min_length", 10, 100, 30)
        max_len = st.number_input("max_length", 100, 500, 130)
        
    if task == "Similarity (SBERT)":
        if not SB_AVAILABLE:
            st.warning("Install sentence-transformers to use this feature.")
        second_sentence = st.text_input("Sentence B", placeholder="Compare to...")
        
    run = st.button("Run")

with right:
    st.header("Output")
    out = st.empty()

if run:
    if task != "Text generation" and not input_text:
        out.error("Please provide input text.")
    else:
        with st.spinner("Processing..."): # Added spinner for UX
            try:
                task_type, model_name = MODEL_MAP[task]

                if task == "Similarity (SBERT)":
                    if not SB_AVAILABLE:
                        out.error("Library missing.")
                    elif not second_sentence:
                        out.error("Provide second sentence.")
                    else:
                        sbert = get_sbert_model() # Use cached loader
                        a = sbert.encode(input_text, convert_to_tensor=True)
                        b = sbert.encode(second_sentence, convert_to_tensor=True)
                        score = util.pytorch_cos_sim(a, b).item()
                        out.metric("Cosine Similarity", f"{score:.4f}")

                elif task == "Text generation":
                    pipe = get_pipeline(task_type, model_name)
                    # FIX: Use max_new_tokens instead of max_length
                    gen = pipe(input_text or "", max_new_tokens=max_new_tokens, do_sample=do_sample)
                    out.write(gen[0]["generated_text"])

                elif task == "Named Entity Recognition (NER)":
                    pipe = get_pipeline(task_type, model_name, aggregation_strategy="simple")
                    entities = pipe(input_text)
                    out.write(entities)

                elif task == "Question Answering (QA)":
                    if not question:
                        out.error("Enter a question.")
                    else:
                        pipe = get_pipeline(task_type, model_name)
                        res = pipe(question=question, context=input_text)
                        out.success(res['answer'])
                        out.json(res)

                elif task == "Summarization":
                    pipe = get_pipeline(task_type, model_name)
                    summ = pipe(input_text, max_length=max_len, min_length=min_len)
                    out.write(summ[0]['summary_text'])

                # Fallback for others
                else:
                    pipe = get_pipeline(task_type, model_name)
                    res = pipe(input_text)
                    out.write(res)

            except Exception as e:
                out.error(f"Error: {str(e)}")
