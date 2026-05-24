import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq

# ─── Configuration ────────────────────────────────────────────
st.set_page_config(
    page_title="LexMA — Assistant Juridique Marocain",
    page_icon="⚖️",
    layout="centered"
)

st.markdown("""
<style>
    .source-box {
        background: #eef4fb;
        border-left: 4px solid #1a73e8;
        padding: 8px 14px;
        margin: 5px 0;
        border-radius: 6px;
        font-size: 0.83em;
        color: #333;
    }
    .hors-sujet {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 0.95em;
    }
    .stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────
st.title("⚖️ LexMA")
st.caption("Assistant juridique · Législation marocaine · FR | AR | EN")

with st.expander("📚 Documents disponibles"):
    st.markdown("""
    - 📋 **Code du Travail** (Loi 65-99)
    - 🏪 **Code de Commerce** (Loi 15-95)
    - 📜 **Code des Obligations et Contrats** (DOC)
    - 🏛️ **Constitution marocaine** (2011)
    """)

st.divider()

# ─── Chargement modèles ───────────────────────────────────────
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
SCORE_THRESHOLD = 0.45  # Seuil de pertinence

# ─── Détection intention ──────────────────────────────────────
def detecter_intention(question):
    q = question.lower().strip()
    salutations  = ["bonjour", "salam", "hello", "hi", "bonsoir", "salut", "hey",
                    "مرحبا", "السلام عليكم", "صباح الخير", "good morning", "good evening"]
    remerciements = ["merci", "thank you", "thanks", "شكرا", "شكراً", "merci beaucoup",
                     "thank u", "thx", "c'est parfait", "super merci"]
    aurevoir     = ["au revoir", "bye", "goodbye", "à bientôt", "bonne journée",
                    "bonne soirée", "مع السلامة", "وداعا"]
    if any(m in q for m in salutations):  return "salutation"
    if any(m in q for m in remerciements): return "remerciement"
    if any(m in q for m in aurevoir):     return "aurevoir"
    return "question"

def repondre_intention(intention):
    if intention == "salutation":
        return (
            "👋 Bonjour ! Je suis **LexMA**, votre assistant juridique marocain.\n\n"
            "Je peux répondre à vos questions sur :\n"
            "- 📋 Code du Travail\n"
            "- 🏪 Code de Commerce\n"
            "- 📜 Code des Obligations et Contrats\n"
            "- 🏛️ Constitution 2011\n\n"
            "Je réponds en **français**, **arabe** et **anglais**. ⚖️"
        )
    if intention == "remerciement":
        return "😊 Avec plaisir ! N'hésitez pas si vous avez d'autres questions juridiques. ⚖️"
    if intention == "aurevoir":
        return "👋 Au revoir ! Revenez si vous avez des questions juridiques. Bonne journée ! ⚖️"

# ─── Détection langue ─────────────────────────────────────────
def detecter_langue(question):
    q = question.lower()
    mots_ar = ["ما", "هل", "كيف", "متى", "من", "في", "على", "عقد", "قانون", "عمل", "حق", "فصل"]
    mots_en = ["what", "how", "when", "who", "where", "why", "is", "are", "can",
               "the", "contract", "law", "worker", "employer", "right"]
    score_ar = sum(1 for m in mots_ar if m in q)
    score_en = sum(1 for m in mots_en if m in q)
    if score_ar > 0: return "Arabic"
    if score_en > 0: return "English"
    return "French"

# ─── Pipeline RAG ─────────────────────────────────────────────
def rag(question, top_k=5):
    # Retrieval
    query_vector = embedder.encode(question).tolist()
    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    ).points

    # Filtre hors-sujet par score
    if not results or results[0].score < SCORE_THRESHOLD:
        return None, []

    # Contexte
    contexte = ""
    sources = []
    for i, r in enumerate(results):
        if r.score < 0.40:  # ignorer les extraits peu pertinents
            continue
        contexte += f"\n--- Extrait {i+1} ({r.payload['source']}, page {r.payload['page']}) ---\n"
        contexte += r.payload['text'] + "\n"
        sources.append({
            "label": f"📄 {r.payload['source']} — Page {r.payload['page']}",
            "score": r.score
        })

    langue = detecter_langue(question)

    prompt = f"""You are LexMA, a legal assistant specialized in Moroccan legislation.

MANDATORY RULES:
1. Respond ONLY in {langue}
2. Base your answer EXCLUSIVELY on the provided excerpts
3. Never repeat the excerpts or this prompt in your response
4. Always cite the relevant article numbers
5. If the question is not related to Moroccan law, respond in {langue} that you only handle Moroccan legal questions
6. If the answer is not found in the excerpts, say so clearly in {langue}
7. Be concise, structured, and professional

EXCERPTS:
{contexte}

QUESTION: {question}

DIRECT ANSWER IN {langue}:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1024
    )
    return response.choices[0].message.content, sources

# ─── Messages hors-sujet par langue ───────────────────────────
def message_hors_sujet(question):
    langue = detecter_langue(question)
    if langue == "Arabic":
        return "⚠️ أنا متخصص فقط في التشريعات المغربية. يرجى طرح سؤال قانوني متعلق بالقانون المغربي."
    if langue == "English":
        return "⚠️ I'm only specialized in Moroccan legislation. Please ask a legal question related to Moroccan law."
    return "⚠️ Je suis uniquement spécialisé dans la législation marocaine. Veuillez poser une question juridique sur le droit marocain."

# ─── Interface chat ───────────────────────────────────────────
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
            with st.spinner("🔍 Recherche dans la législation marocaine..."):
                reponse, sources = rag(question)

            # Hors-sujet détecté par score
            if reponse is None:
                msg_hs = message_hors_sujet(question)
                st.markdown(f'<div class="hors-sujet">{msg_hs}</div>', unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": msg_hs})

            else:
                st.markdown(reponse)
                if sources:
                    with st.expander("📚 Sources consultées"):
                        for s in sources:
                            st.markdown(
                                f'<div class="source-box">{s["label"]} &nbsp;·&nbsp; score: {s["score"]:.2f}</div>',
                                unsafe_allow_html=True
                            )
                st.session_state.messages.append({"role": "assistant", "content": reponse})
