from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

schemes = [
    {
        "id": "pension",
        "title": "వృద్ధాప్య పెన్షన్",
        "description": "60 సంవత్సరాలు పైబడిన వారికి ఆర్థిక సహాయం",
        "min_age": 60
    },
    {
        "id": "scholarship",
        "title": "విద్యార్థి ఉపకార వేతనం",
        "description": "పాఠశాల మరియు కాలేజీ విద్యార్థులకు సహాయం",
        "min_age": 10,
        "max_age": 25
    }
]

embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
texts = [s["title"] + " " + s["description"] for s in schemes]
embeddings = embedder.encode(texts, convert_to_numpy=True)
faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

def retrieve_scheme(query):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, 1)
    return schemes[I[0][0]]

def check_eligibility(age, scheme):
    if age is None:
        return False, "మీ వయస్సు తెలియదు"
    if age < scheme.get("min_age", 0):
        return False, f"కనీస వయస్సు {scheme['min_age']} కావాలి"
    if "max_age" in scheme and age > scheme["max_age"]:
        return False, "వయస్సు పరిమితిని మించిపోయింది"
    return True, "మీరు అర్హులు"
import gradio as gr

def gradio_agent(query, age):
    scheme = retrieve_scheme(query)
    eligible, message = check_eligibility(age, scheme)
    return f"Scheme: {scheme['title']}\nEligibility: {message}"


demo = gr.Interface(
    fn=gradio_agent,
    inputs=[
        gr.Textbox(label="Enter your query"),
        gr.Number(label="Enter your age")
    ],
    outputs=gr.Textbox(label="Result"),
    title="CredResolve AI Agent"
)

demo.launch(share=True)
