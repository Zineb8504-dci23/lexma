import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq

st.set_page_config(
    page_title="LexMA — Assistant Juridique Marocain",
    page_icon="⚖️",
    layout="centered"
)

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

st.title("⚖️ LexMA")
st.caption("Assistant juridique basé sur la législation marocaine — FR | AR | EN")
st.divider()

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

def detecter_intention(question):
    q = question.lower().strip()
    
    salutations = ["bonjour", "salam", "hello", "hi", "bonsoir", "salut", "hey",
                   "مرحبا", "السلام عليكم", "صباح الخير", "good morning", "good evening"]
    
    remerciements = ["merci", "thank you", "thanks", "شكرا", "شكراً", "merci beaucoup",
                     "thank u", "thx", "je te remercie", "c'est parfait", "super", "excellent"]
    
    aurevoir = ["au revoir", "bye", "goodbye", "à bientôt", "bonne journée",
                "bonne soirée", "مع السلامة", "وداعا"]
    
    if any(mot in q for mot in salutations):
        return "salutation"
    if any(mot in q for mot in remerciements):
        return "remerciement"
    if any(mot in q for mot in aurevoir):
        return "aurevoir"
    return "question"

def repondre_intention(intention):
    if intention == "salutation":
        return "👋 Bonjour ! Je suis **LexMA**, votre assistant juridique marocain.\n\nPosez-moi vos questions sur le **Code du Travail**, le **Code de Commerce**, le **Code des Obligations et Contrats** ou la **Constitution**.\n\nJe réponds en **français**, **arabe** et **anglais** ! ⚖️"
    
    if intention == "remerciement":
        return "😊 Avec plaisir ! N'hésitez pas si vous avez d'autres questions juridiques. Je suis là pour vous aider ! ⚖️"
    
    if intention == "aurevoir":
        return "👋 Au revoir ! N'hésitez pas à revenir si vous avez des questions juridiques. Bonne journée ! ⚖️"

def detecter_langue(question):
    q = question.lower()
    mots_ar = ["ما", "هل", "كيف", "متى", "من", "في", "على", "عقد", "قانون", "عمل"]
    mots_en = ["what", "how", "when", "who", "where", "why", "is", "are", "can", "the", "contract", "law"]
    
    score_ar = sum(1 for mot in mots_ar if mot in q)
    score_en = sum(1 for mot in mots_en if mot in q)
    
    if score_ar > 0:
        return "Arabic"
    if score_en > 0:
        return "English"
    return "French"

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

    langue = detecter_langue(question)

    prompt = f"""You are LexMA, a legal assistant specialized in Moroccan law.

YOU MUST RESPOND IN {langue} ONLY. THIS IS MANDATORY.

Rules:
- Answer based ONLY on the provided legal excerpts
- Never repeat the excerpts or the prompt
- Give only your final answer
- If the answer is not in the excerpts, say so in {langue}
- Always cite relevant articles

EXCERPTS:
{contexte}

QUESTION: {question}

ANSWER IN {langue}:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1024
    )
    return response.choices[0].message.content, sources

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Posez votre question juridique... (FR | AR | EN)"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        intention = detecter_intention(question)
        
        if intention != "question":
            reponse = repondre_intention(intention)
            st.markdown(reponse)
            st.session_state.messages.append({"role": "assistant", "content": reponse})
        else:
            with st.spinner("Recherche dans la législation marocaine..."):
                reponse, sources = rag(question)
            st.markdown(reponse)
            with st.expander("📚 Sources consultées"):
                for s in sources:
                    st.markdown(f'<div class="source-box">{s}</div>', unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": reponse})
