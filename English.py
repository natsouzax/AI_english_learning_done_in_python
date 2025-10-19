import pandas as pd
import random
import streamlit as st
import nltk
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import os
import html 

# --- 1. NLTK Configuration (Ensure WordNet) ---
if 'wordnet_checked' not in st.session_state:
    st.session_state['wordnet_checked'] = True 
    with st.spinner("Checking and downloading NLTK resources (WordNet)..."):
        try:
            nltk.download('wordnet', quiet=True)
        except Exception as e:
            st.error(f"Failed to download WordNet: {e}")
            st.stop()


# --- 2. Page Configuration and Styles (INVERTED PALETTE) ---
st.set_page_config(
    page_title="Review.IA – Flashcards Quizlet Style",
    layout="centered"
)

# Palette: Main Blue (#4255ff), Background Gray (#f7f9fa), Border/Detail Gray (#e0e6ed), Black/Dark (#1a1a1a)
st.markdown("""
<style>
/* 1. Streamlit Elements Colors and Layout */
:root {
    --primary-color: #4255ff;
}
/* Style for quiz options */
div.stRadio > label:nth-child(n) {
    background-color: #f0f0f0; 
    border-radius: 6px;
    padding: 10px;
    margin-bottom: 5px;
}

/* Increase space between the subtitle (Flashcard: Word) and the card */
h3.st-emotion-cache-1czwkmk { /* Common selector for st.subheader in Streamlit */
    margin-bottom: 2.5rem; 
}


/* 2. 3D Rotation Effect (FLIP CARD) */
.flip-container {
    perspective: 10000px;
    width: 100%;
    min-height: 250px; 
    margin-bottom: 10px; 
}

.flipper {
    transition: 0.6s;
    transform-style: preserve-3d;
    position: relative;
    height: 100%; 
}

/* "Flipped" State */
.flip-container.flipped .flipper {
    transform: rotateY(180deg);
}

/* Card Sides */
.flashcard-side {
    backface-visibility: hidden;
    position: absolute;
    top: 0;
    left: 0;
    width: 100%; 
    min-height: 250px;
    height: 100%; 
    
    /* Card Visual Styles (Unified) */
    border: 1px solid #e0e6ed;
    border-radius: 16px; 
    padding: 50px 40px; 
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
}

.front {
    background-color: #ffffff;
    z-index: 2;
    transform: rotateY(0deg);
}

.back {
    background-color: #f0f4ff; /* Kept as light blue for back contrast */
    transform: rotateY(180deg);
}

/* Rotation Correction: Un-flips the back content (IMPORTANT for readability) */
.flip-container.flipped .back {
    transform: rotateY(180deg);
}
.flip-container.flipped .back p, 
.flip-container.flipped .back hr,
.flip-container.flipped .back .st-emotion-cache-1l0353 {
    transform: rotateY(0deg); 
}


/* 3. Content Styles (INVERTED COLORS) */

/* Main title (h1) IS NOW BLACK */
h1 { color: #1a1a1a; } 

/* Subtitles (h2, h3) ARE NOW BLUE */
h2, h3 { color: #4255ff; }

/* Word Title on Card (h2) IS NOW BLUE */
.flashcard-side h2 {
    color: #4255ff;
    font-size: 2.8em; 
    margin-bottom: 0.6em;
    text-transform: uppercase;
}
.flashcard-side p {
    color: #4a5568; /* Kept gray for standard body text */
    font-size: 1.3em; 
    line-height: 1.6;
}


/* 4. Feedback Colors */
.stSuccess {
    background-color: #f0f4ff;
    /* SUCCESS BORDER IS NOW BLACK */
    border-left: 5px solid #1a1a1a; 
}
.stError {
    background-color: #f7f9fa;
    border-left: 5px solid #8e949d;
}
</style>
""", unsafe_allow_html=True)


# --- 3. Resource Loading (Cache) ---

@st.cache_resource
def load_nlp_resources():
    fill_mask = pipeline("fill-mask", model="distilbert-base-uncased")
    definition_generator = pipeline("text2text-generation", model="google/flan-t5-base")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return fill_mask, definition_generator, embedder

@st.cache_resource
def load_data():
    if not os.path.exists("10000_Words.csv") or not os.path.exists("templates.csv"):
        # Simulated data for testing, in case CSV files do not exist
        base_words = ["word", "test", "example", "definition", "streamlit"]
        templates = ["The {palavra_alvo} is a good [MASK] of this rule.", "We should always [MASK] the {palavra_alvo} before using it."]
        st.warning("Files '10000_Words.csv' or 'templates.csv' not found. Using test data.")
        return base_words, templates
        
    df_words = pd.read_csv("10000_Words.csv", header=None)
    base_words = df_words[0].tolist()

    templates_df = pd.read_csv("templates.csv", skiprows=1, header=None)
    templates = templates_df[0].tolist()
    
    return base_words, templates


# --- 4. Initialize Resources and Functions ---
try:
    fill_mask, definition_generator, embedder = load_nlp_resources()
except Exception:
    st.error("Failed to load NLP models. The application may not function correctly.")
    # Mock functions to prevent breaks if the API fails to load
    def fill_mask(*args, **kwargs):
        return [{"sequence": "This is a [MASK] example sentence.", "score": 1.0}]
    def definition_generator(*args, **kwargs):
        return [{"generated_text": "A simulated definition."}]
    
    class MockEmbedder:
        def encode(self, text): return [0]
    embedder = MockEmbedder()

base_words, templates = load_data()

# --- The Flashcard Functions (Logic) ---

def generate_example(word):
    template = random.choice(templates)
    
    if "[MASK]" not in template:
        return {"sentence": "Invalid example template.", "score": 0.0, "template": template}

    sentence_with_mask = template.replace("{palavra_alvo}", word) # Note: '{palavra_alvo}' kept from original logic
    
    with st.spinner('Generating usage example...'):
        try:
            result = fill_mask(sentence_with_mask, top_k=1)[0]
            # Ensure the word is inserted where the mask was
            sentence = result["sequence"].replace("[MASK]", word) 
            return {
                "sentence": sentence, 
                "score": result["score"],
                "template": template
            }
        except Exception:
            return {"sentence": f"Error generating example for '{word}'.", "score": 0.0, "template": template}

def _synset_freq(ss):
    return sum(l.count() for l in ss.lemmas())

def generate_definition(word, pos_hint=None):
    pos_map = {'n': wn.NOUN, 'v': wn.VERB, 'a': wn.ADJ, 'r': wn.ADV, None: None}
    ss = wn.synsets(word, pos=pos_map.get(pos_hint, None))

    if ss:
        ss_sorted = sorted(ss, key=_synset_freq, reverse=True)
        defs, seen = [], set()
        for s in ss_sorted:
            d = s.definition().strip()
            if d not in seen:
                defs.append(d)
                seen.add(d)
            if len(defs) == 2:
                break
        if defs:
            return defs[0]

    # Fallback to Flan-T5
    prompt = (
        f"Give a concise, dictionary-style definition of the English word '{word}'. "
        "5–15 words. No examples, no comparisons."
    )
    with st.spinner('Generating definition with Flan-T5...'):
        try:
            resp = definition_generator(prompt, max_length=32, do_sample=False)[0]["generated_text"].strip()
            return resp
        except Exception:
            return f"No definition found for {word}."

def generate_synonyms_antonyms(word, top_n=5, similarity_threshold=0.55):
    # Synonyms/Antonyms Logic (depends on embedder and WordNet)
    try:
        with st.spinner('Searching for synonyms and antonyms...'):
            word_embedding = embedder.encode(word)
            wn_synonyms = set()
            wn_antonyms = set()

            for pos in [wn.NOUN, wn.VERB, wn.ADJ, wn.ADV]:
                for syn in wn.synsets(word, pos=pos):
                    for lemma in syn.lemmas():
                        wn_synonyms.add(lemma.name().replace('_', ' '))
                        if lemma.antonyms():
                            wn_antonyms.add(lemma.antonyms()[0].name().replace('_', ' '))

            filtered_syn = []
            for s in wn_synonyms:
                score = util.cos_sim(word_embedding, embedder.encode(s)).item()
                if score > similarity_threshold and s.lower() != word.lower():
                    filtered_syn.append((s, score))

            filtered_ant = []
            for a in wn_antonyms:
                score = util.cos_sim(word_embedding, embedder.encode(a)).item()
                if score < similarity_threshold:
                    filtered_ant.append((a, score))

            filtered_syn = sorted(filtered_syn, key=lambda x: x[1], reverse=True)
            filtered_ant = sorted(filtered_ant, key=lambda x: x[1])

            synonyms = [s for s,_ in filtered_syn[:top_n]]
            antonyms = [a for a,_ in filtered_ant[:top_n]]

            return synonyms, antonyms
    except Exception:
        return [], []

def rephrase_definition(definition, word):
    prompt = (
        f"Rewrite the definition of the word '{word}' in a different way, "
        f"using synonyms and a complete sentence, but keep the same meaning.\n\n"
        f"Definition: {definition}"
    )
    try:
        resp = definition_generator(prompt, max_length=80, do_sample=False)[0]["generated_text"].strip()
        if resp and resp.lower() != definition.lower():
            return resp
        return f"It refers to {definition}"
    except Exception:
        return f"It refers to {definition}"

def generate_quiz(word, definition):
    question = f"What does **'{word}'** mean?"
    correct = rephrase_definition(definition, word)
    
    # Pool of fake answers (Kept the original long list for variety)
    fake_answers = [
        "A type of plant found in the Amazon.", "A person who dislikes music.", 
        "A rare type of mineral.", "An old-fashioned word for 'blue'.", 
        "A traditional dance from Eastern Europe.", "A forgotten constellation in the night sky.", 
        "A small tool used by blacksmiths.", "An ancient form of currency.", 
        "A medieval board game similar to chess.", "A mythical bird from Greek folklore.", 
        "A special type of ink used in ancient manuscripts.", "An old medical treatment involving herbs.", 
        "A ceremonial hat worn by priests.", "A style of poetry from the 15th century.", 
        "An ancient unit of measurement.", "A secret code used by sailors.", 
        "A type of dessert popular in the 18th century.", "A lost language spoken by traders.", 
        "An ornamental stone used in royal crowns.", "A nickname for a type of storm cloud.", 
        "A festival once held in Mesopotamia.", "A rare color pigment used by Renaissance painters.", 
        "A kind of armor used only in tournaments.", "A forgotten musical instrument made of glass.", 
        "A ritual practiced by ancient farmers.", "A hairstyle worn by Egyptian nobles.", 
        "An ancient recipe for preserving meat.", "A charm believed to ward off evil spirits.", 
        "A secret society from the 12th century.", "A type of bread baked only during winter.", 
        "A decorative knot used by sailors.", "A hunting tool invented in the Bronze Age.", 
        "A title given to storytellers in medieval courts.", "A traditional mask used in rituals.", 
        "An extinct species of domesticated animal.", "A mathematical symbol no longer in use.", 
        "A piece of jewelry worn by warriors.", "A children’s game played in ancient Rome.", 
        "A word once used to describe thunderstorms.", "A type of scroll used in early libraries.", 
        "An honorary rank in old universities.", "A special fabric woven by monks.", 
        "A mythical plant said to glow at night.", "A healing ritual involving fire and water.", 
        "A gemstone believed to bring wisdom.", "A forgotten military strategy.", 
        "A unit of time used by Mayan priests.", "A seasonal song sung by shepherds.", 
        "A charm carved from animal bones.", "A type of drink made com honey and spices.", 
        "A forgotten symbol of good luck.", "An honorary title for traveling merchants.", 
        "A ritual performed during eclipses.", "A lost form of early photography.", 
        "A rare flower used in coronations.", "A fishing tool made of bone.", 
        "A unit of distance used by Vikings.", "A forgotten musical scale.", 
        "A hat worn only by judges in medieval Europe.", "A medicinal root once traded like gold.", 
        "A color once forbidden to commoners.", "A dessert once reserved for kings.", 
        "A poem written without vowels.", "A holiday celebrated by lighting rivers on fire.", 
        "A forgotten sport played com stones.", "A flute carved from ivory.", 
        "A cloak used in coronation ceremonies.", "A symbol used in early maps to mark treasure.", 
        "A tool used by alchemists.", "A toy made of clay used by Greek children.", 
        "A hairstyle reserved for warriors.", "A crown made entirely of feathers.", 
        "A secret recipe for eternal youth.", "A forgotten constellation shaped like a wolf.", 
        "A musical instrument played com fire.", "A lamp that never went out.", 
        "A ring believed to control the tides.", "A charm that made crops grow faster.", 
        "A forgotten recipe for glassmaking.", "A special coin used to pay storytellers.", 
        "A ladder made of silk used in rituals.", "A throne made of bronze.", 
        "A forgotten alphabet com 50 letters.", "A necklace worn only durante eclipses.", 
        "A festival where people wore masks of animals.", "A drum used to call spirits.", 
        "A map drawn on animal skin.", "A recipe for ink that glowed in the dark.", 
        "A bird believed to bring storms.", "A staff used by village elders.", 
        "A forgotten empire of the desert.", "A book bound in silver.", 
        "A medicine made from volcanic ash.", "A flute made of gold.", 
        "A bridge built from tree roots.", "A forgotten city under the sea.", 
        "A type of bread shaped like stars.", "A blanket woven to tell stories.", 
        "A shield made of glass.", "A necklace believed to protect sailors.", 
        "A forgotten dance of the harvest.", "A candle that burned underwater.", 
        "A jewel said to trap sunlight.", "A crown made of seashells.", 
        "A recipe for making invisible ink.", "A ritual where animals were painted gold.", 
        "A type of fabric softer than silk.", "A drink that glowed in the dark.", 
        "A tower built to watch the stars.", "A statue that sang com the wind.", 
        "A mask used to speak to gods.", "A horn that summoned rain.", 
        "A boat made entirely of paper.", "A forgotten trade route across the ocean.", 
        "A stone that changed color com the moon.", "A ladder carved into mountains.", 
        "A hidden valley of eternal spring.", "A lost temple of mirrors.", 
        "A forgotten feast of the harvest moon.", "A weapon shaped like a crescent.", 
        "A forgotten ritual of silence.", "A key said to open any lock.", 
        "A tower that disappeared at dawn.", "A coin that always returned to its owner.", 
        "A book that could not be burned.", "A harp that played itself at night.", 
        "A forgotten river of silver.", "A cloak that made the wearer invisible.", 
        "A bell that summoned birds.", "A forest where trees glowed in the dark.", 
        "A cave filled com crystals that sang.", "A forgotten bridge of light.", 
        "A pearl that whispered secrets.", "A forgotten throne of ice.", 
        "A ship that sailed without wind.", "A lantern that floated in the air.", 
        "A forgotten gate to the stars.", "A key shaped like the sun.", 
        "A flute that summoned animals.", "A stone path that moved at night.", 
        "A forgotten village of glass houses.", "A mirror that showed the future.", 
        "A lake that never froze.", "A bird com feathers of fire.", 
        "A mountain said to be hollow.", "A forgotten crown of thorns.", 
        "A bracelet that grew tighter com lies.", "A mask that made the wearer fearless.", 
        "A tree that bloomed once a century.", "A staff that glowed under the moon.", 
        "A forgotten city of gold.", "A ring that whispered names.", 
        "A sword that could cut shadows.", "A festival where fire never burned.", 
        "A forgotten spell of endless sleep.", "A painting that moved at night.", 
        "A door that opened to nowhere.", "A candle that revealed secrets.", 
        "A lake that reflected another world.", "A flower that never wilted.", 
        "A star that fell every year.", "A song sung only by the wind.", 
        "A forgotten language of whistles.", "A mask carved from a single bone.", 
        "A forgotten oath of silence.", "A necklace that hummed com energy.", 
        "A forgotten map drawn in blood.", "A bird that sang at midnight only.", 
        "A forgotten crown of leaves.", "A staff carved from a lightning bolt.", 
        "A hidden garden under the desert.", "A forgotten shipwreck of silver coins.", 
        "A lantern that guided lost souls.", "A robe worn only during full moons.", 
        "A forgotten jewel of the desert.", "A harp strung com spider silk.", 
        "A forgotten story told by shadows.", "A festival where no one spoke.", 
        "A forgotten road paved com gold.", "A mask of mirrors that blinded enemies.", 
        "A forgotten hammer that built temples.", "A crown made of frozen water.", 
        "A forgotten language of drums.", "A flute that made snakes dance.", 
        "A forgotten drink of immortality."
    ]
        
    options = [correct] + random.sample(fake_answers, 3) 
    random.shuffle(options)
    
    if correct not in options:
        options[random.randint(0, 3)] = correct

    return {"question": question, "options": options, "answer": correct}

def generate_flashcard(word):
    with st.empty(): 
        with st.spinner(f"Generating Flashcard for '{word}'..."):
            example = generate_example(word)
            definition = generate_definition(word)
            synonyms, antonyms = generate_synonyms_antonyms(word)
            quiz = generate_quiz(word, definition)
            
            return {
                "word": word,
                "example_sentence": example["sentence"],
                "definition": definition,
                "synonyms": synonyms,
                "antonyms": antonyms,
                "quiz": quiz
            }

# --- 5. Streamlit Interface Logic (Quizlet Style) ---

st.title(" 🧠 Review.IA – Flashcards ")

if "card_flipped" not in st.session_state:
    st.session_state["card_flipped"] = False

# FUNCTION: Flip Card and Rerun
def flip_card():
    st.session_state["card_flipped"] = not st.session_state["card_flipped"]
    st.rerun() 

# --- SIDEBAR: Input and Controls ---
with st.sidebar:
    st.header("⚙️ Controls")
    word_default = st.session_state.get("flashcard", {}).get("word", "curious")
    word = st.text_input("Enter an English word:", key="word_input_sb", value=word_default)
    
    if st.button("Generate Flashcard", use_container_width=True, type="primary") and word:
        if word.strip():
            st.session_state["flashcard"] = generate_flashcard(word.strip())
            st.session_state["quiz_answered"] = False
            st.session_state["quiz_feedback"] = ""
            st.session_state["card_flipped"] = False 
            st.rerun()


# --- MAIN BODY: Interactive Card ---
if "flashcard" in st.session_state:
    flashcard = st.session_state["flashcard"]
    
    # 1. MAIN CARD (Click to Flip)
    st.subheader(f"🎴 Flashcard: {flashcard['word'].upper()}")
    
    flip_class = "flipped" if st.session_state["card_flipped"] else ""
    
    # HTML DECODING: Decodes and uses .join for HTML stability
    decoded_definition = html.unescape(flashcard['definition'])
    decoded_example = html.unescape(flashcard['example_sentence'])

    card_html_parts = [
        f'<div class="flip-container {flip_class}">',
        '    <div class="flipper">',
        # FRENTE DO CARD
        '        <div class="front flashcard-side">',
        # H2 no CSS agora é AZUL, mas queremos o texto de instrução PRETO/ESCURO
        f'            <h2>{flashcard["word"].upper()}</h2>', 
        f'            <p style="color:#1a1a1a; font-weight:bold;">Clique no botão abaixo para ver a definição.</p>',
        '        </div>',
        # VERSO DO CARD (CORES INVERTIDAS)
        '        <div class="back flashcard-side">',
        # Título "Definição" AGORA é PRETO
        f'            <p style="color:#1a1a1a; font-size: 1.5em; font-weight:bold;">📚 Definition</p>',
        # Corpo da Definição AGORA é AZUL
        f'            <p style="color:#4255ff;">{decoded_definition}</p>',
        # Separador AGORA é PRETO
        '            <hr style="border-top: 1px solid #1a1a1a; width: 50%;">', 
        f'            <p style="font-style: italic; color: #4a5568;">📝 Exemple: {decoded_example}</p>',
        '        </div>',
        '    </div>',
        '</div>'
    ]
    
    card_html = "\n".join(card_html_parts)

    # Renders the HTML structure
    st.markdown(card_html, unsafe_allow_html=True)
    
    # Streamlit button that triggers the flip_card function
    if st.button("Flip Card", key="flip_button", use_container_width=True):
        flip_card()
    
    
    st.markdown("---")


    # 2. DETAILS AND RELATED INFORMATION
    st.subheader("💡 Related Information")
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"**🟩 Synonyms:** {', '.join(flashcard['synonyms']) if flashcard['synonyms'] else 'None found.'}")
    with col2:
        st.error(f"**🟥 Antonyms:** {', '.join(flashcard['antonyms']) if flashcard['antonyms'] else 'None found.'}")
    
    st.markdown("---")


    # 3. MULTIPLE CHOICE QUIZ
    st.subheader("📝 Test Your Knowledge")
    quiz = flashcard["quiz"]
    
    st.markdown(f"**❓ Question:** {quiz['question']}")
    
    if not st.session_state.get("quiz_answered", False):
        
        with st.form(key='quiz_form'):
            selected_option = st.radio("Choose the correct option:", quiz["options"], key="radio_quiz_options")
            submit_button = st.form_submit_button(label='Answer')

            if submit_button:
                chosen_answer = selected_option
                if chosen_answer == quiz["answer"]:
                    st.session_state["quiz_feedback"] = "🎉 **Correct!** Excellent work."
                    st.session_state["quiz_answered"] = True
                else:
                    st.session_state["quiz_feedback"] = f"❌ **Wrong.** The correct answer was: **{quiz['answer']}**"
                    st.session_state["quiz_answered"] = True
                st.rerun()

    # Feedback Block and Action Buttons
    if st.session_state.get("quiz_answered", False):
        st.markdown(st.session_state["quiz_feedback"])
        
        col_quiz_new, col_clear = st.columns(2)
        
        with col_quiz_new:
            if st.button("🔄 Generate New Quiz", use_container_width=True):
                new_quiz = generate_quiz(flashcard["word"], flashcard["definition"])
                st.session_state["flashcard"]["quiz"] = new_quiz
                st.session_state["quiz_answered"] = False
                st.session_state["quiz_feedback"] = ""
                st.rerun()

        with col_clear:
            if st.button("🗑️ Clear and Try Another Word", use_container_width=True, type="primary"):
                st.session_state.pop("flashcard", None)
                st.session_state["quiz_answered"] = False
                st.session_state["quiz_feedback"] = ""
                st.session_state["card_flipped"] = False
                st.rerun()

        st.markdown("---")
        st.info("To generate a flashcard for a different word, use the **⚙️ Controls** sidebar.")