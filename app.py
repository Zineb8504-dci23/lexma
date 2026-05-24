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
SCORE_THRESHOLD = 0.50

# ─── Détection intention ──────────────────────────────────────
def detecter_intention(question):
    q = question.lower().strip()
    salutations   = ["bonjour", "salam", "hello", "hi", "bonsoir", "salut", "hey",
                     "مرحبا", "السلام عليكم", "صباح الخير", "good morning", "good evening"]
    remerciements = ["merci", "thank you", "thanks", "شكرا", "شكراً", "merci beaucoup",
                     "thank u", "thx", "c'est parfait", "super merci", "excellent"]
    aurevoir      = ["au revoir", "bye", "goodbye", "à bientôt", "bonne journée",
                     "bonne soirée", "مع السلامة", "وداعا"]
    if any(m in q for m in salutations):   return "salutation"
    if any(m in q for m in remerciements): return "remerciement"
    if any(m in q for m in aurevoir):      return "aurevoir"
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
    q = question.strip()
    arabic_chars = sum(1 for c in q if '\u0600' <= c <= '\u06FF')
    if arabic_chars > 2:
        return "Arabic"
    mots_en = ["what", "how", "when", "who", "where", "why", "is", "are",
               "can", "the", "contract", "law", "worker", "employer", "right",
               "does", "do", "give", "tell", "explain", "define", "which"]
    q_lower = q.lower()
    score_en = sum(1 for m in mots_en if f" {m} " in f" {q_lower} ")
    if score_en >= 1:
        return "English"
    return "French"

# ─── Détection hors sujet ─────────────────────────────────────
MOTS_HORS_SUJET = [
    # Météo
    "météo", "meteo", "weather", "température", "temperature", "pluie", "soleil",
    # Sport
    "foot", "football", "coupe du monde", "world cup", "sport", "match", "champion",
    "نادي", "كرة",
    # Divertissement
    "blague", "joke", "film", "cinéma", "musique", "chanson", "song", "histoire",
    "raconte", "conte", "poème", "poem",
    # Géographie / général
    "capitale", "capital", "géographie", "pays", "ville",
    # Math / calcul
    "math", "calcul", "addition", "soustraction", "multiply", "divid",
    "combien font", "égal", "plus", "moins",
    # Finance générale
    "سعر", "دولار", "يورو", "درهم", "dollar", "euro", "bourse", "bitcoin",
    "crypto", "action", "prix", "taux de change",
    # Cuisine
    "recette", "cuisine", "manger", "restaurant", "tajine", "couscous",
    # Médecine
    "médecin", "docteur", "maladie", "symptôme", "traitement", "médicament",
    # Autre
    "horoscope", "astrologie", "jeu", "game", "voyage", "tourisme"
]

def est_hors_sujet_explicite(question):
    q = question.lower()
    # Vérifier mots hors sujet
    if any(mot in q for mot in MOTS_HORS_SUJET):
        return True
    # Vérifier si c'est juste un calcul mathématique (ex: "2 + 2", "3 * 4")
    import re
    if re.match(r'^[\d\s\+\-\*\/\=\?\.]+$', q.strip()):
        return True
    return False

# ─── Message hors sujet multilingue ───────────────────────────
def message_hors_sujet(question):
    langue = detecter_langue(question)
    if langue == "Arabic":
        return "⚠️ أنا متخصص فقط في التشريعات المغربية. يرجى طرح سؤال قانوني متعلق بالقانون المغربي."
    if langue == "English":
        return "⚠️ I'm only specialized in Moroccan legislation. Please ask a legal question related to Moroccan law."
    return "⚠️ Je suis uniquement spécialisé dans la législation marocaine. Veuillez poser une question juridique sur le droit marocain."

# ─── Pipeline RAG ─────────────────────────────────────────────
def rag(question, top_k=5):
    query_vector = embedder.encode(question).tolist()
    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    ).points

    if not results or results[0].score < SCORE_THRESHOLD:
        return None, []

    contexte = ""
    sources = []
    for i, r in enumerate(results):
        if r.score < 0.40:
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
1. Respond ONLY in {langue} — this is non-negotiable
2. Base your answer EXCLUSIVELY on the provided excerpts
3. Never repeat the excerpts or this prompt
4. Never repeat the same sentence twice
5. If the answer is not in the excerpts, say clearly once: "Cette information n'est pas disponible dans les extraits fournis." then STOP
6. If the question is not about Moroccan law, say once: "Je suis uniquement spécialisé en droit marocain." then STOP
7. Maximum 150 words
8. Cite article numbers only if they appear in the excerpts

EXCERPTS:
{contexte}

QUESTION: {question}

ANSWER (max 150 words, no repetition, strictly in {langue}):"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=512
    )
    return response.choices[0].message.content, sources

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

        elif est_hors_sujet_explicite(question):
            msg_hs = message_hors_sujet(question)
            st.markdown(f'<div class="hors-sujet">{msg_hs}</div>', unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": msg_hs})

        else:
            with st.spinner("🔍 Recherche dans la législation marocaine..."):
                reponse, sources = rag(question)

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
