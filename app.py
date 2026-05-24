import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq

# ─── Configuration page ───────────────────────────────────────
st.set_page_config(
    page_title="LexMA — Assistant Juridique Marocain",
    page_icon="⚖️",
    layout="centered"
)

# ─── CSS personnalisé ─────────────────────────────────────────
st.markdown("""
<style>
    .source-box {
        background: #e8f4f8;
        border-left: 4px solid #1a73e8;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 4px;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────
st.title("⚖️ LexMA")
st.caption("Assistant juridique basé sur la législation marocaine — FR | AR | EN")
st.divider()

# ─── Chargement des modèles ───────────────────────────────────
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("BAAI/bge-m3")
    qdrant = QdrantClient(
        url=st.secrets["QDRANT_URL"],
        api_key=st.secrets["QDRANT_API_KEY"]
    )
    groq = Groq(api_key=st.secrets["GROQ_API_KEY"])
    return embedder, qdrant, groq

embedder, qdrant_client, groq_client = load_models()

COLLECTION_NAME = "lexma_juridique"

# ─── Fonction RAG ─────────────────────────────────────────────
def rag(question, top_k=5):
    query_vector = embedder.encode(question).tolist()
    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    ).points

    contexte = ""
    sources = []
    for i, r in enumerate(results):
        contexte += f"\n--- Extrait {i+1} ({r.payload['source']}, page {r.payload['page']}) ---\n"
        contexte += r.payload['text'] + "\n"
        sources.append(f"📄 {r.payload['source']} — Page {r.payload['page']} (score: {r.score:.2f})")

    prompt = f"""Tu es LexMA, un assistant juridique spécialisé dans la législation marocaine.

RÈGLE ABSOLUE : Réponds OBLIGATOIREMENT dans la même langue que la question.
- Question en français → réponse en français
- Question en arabe → réponse en arabe
- Question en anglais → réponse en anglais

Réponds en te basant UNIQUEMENT sur les extraits juridiques fournis.
Si la réponse n'est pas dans les extraits, dis-le clairement.
Cite toujours la source (nom du code + numéro d'article).

EXTRAITS :
{contexte}

QUESTION : {question}

RÉPONSE (obligatoirement dans la langue de la question) :"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1024
    )

    return response.choices[0].message.content, sources

# ─── Historique ───────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─── Input ────────────────────────────────────────────────────
if question := st.chat_input("Posez votre question juridique... (FR | AR | EN)"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche dans la législation marocaine..."):
            reponse, sources = rag(question)

        st.markdown(reponse)

        with st.expander("📚 Sources consultées"):
            for s in sources:
                st.markdown(f'<div class="source-box">{s}</div>', unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": reponse})