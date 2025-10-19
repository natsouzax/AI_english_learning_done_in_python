import pandas as pd
import random
import streamlit as st
import nltk
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import os
import html 
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer # Adicionado para o novo modelo de tradução

# --- 1. Configuração do NLTK (Garantindo WordNet) ---
if 'wordnet_checked' not in st.session_state:
    st.session_state['wordnet_checked'] = True 
    with st.spinner("Verificando e baixando recursos do NLTK (WordNet)..."):
        try:
            nltk.download('wordnet', quiet=True)
        except Exception as e:
            st.error(f"Falha ao baixar WordNet: {e}")
            st.stop()


# --- 2. Configuração da Página e Estilos (Mantidos) ---
st.set_page_config(
    page_title="Review.IA – Flashcards Quizlet Style",
    layout="centered"
)

# Paleta: Azul Principal (#4255ff), Cinza Fundo (#f7f9fa), Cinza Borda/Detalhe (#e0e6ed), Preto/Escuro (#1a1a1a)
st.markdown("""
<style>
/* 1. Cores e Layout de Elementos Streamlit */
:root {
    --primary-color: #4255ff;
}
/* Estilo para as opções de quiz */
div.stRadio > label:nth-child(n) {
    background-color: #f0f0f0; 
    border-radius: 6px;
    padding: 10px;
    margin-bottom: 5px;
}

/* Aumenta o espaço entre o subtítulo (Flashcard: Palavra) e o card */
h3.st-emotion-cache-1czwkmk { /* Seletor comum para st.subheader no Streamlit */
    margin-bottom: 2.5rem; 
}


/* 2. Efeito de Rotação 3D (FLIP CARD) */
.flip-container {
    perspective: 10000px;
    width: 100%;
    /* CORREÇÃO DE OVERFLOW: Aumentado para dar mais espaço de base */
    min-height: 300px; 
    margin-bottom: 10px; 
}

.flipper {
    transition: 0.6s;
    transform-style: preserve-3d;
    position: relative;
    height: 100%; 
}

/* Estado de "Virado" */
.flip-container.flipped .flipper {
    transform: rotateY(180deg);
}

/* Lados do Card */
.flashcard-side {
    backface-visibility: hidden;
    position: absolute;
    top: 0;
    left: 0;
    width: 100%; 
    /* CORREÇÃO DE OVERFLOW: Aumentado para dar mais espaço de base */
    min-height: 300px; 
    height: 100%; 
    
    /* Estilos Visuais do Card (Unificado) */
    border: 1px solid #e0e6ed;
    border-radius: 16px; 
    padding: 50px 40px; 
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    
    /* CORREÇÃO CRÍTICA DE OVERFLOW/QUEBRA DE LINHA */
    overflow-wrap: break-word; 
    word-break: break-word;
}

.front {
    background-color: #ffffff;
    z-index: 2;
    transform: rotateY(0deg);
}

.back {
    background-color: #f0f4ff; /* Mantido como azul claro para contraste no verso */
    transform: rotateY(180deg);
}

/* Correção de Giro: Desvira o conteúdo do verso (IMPORTANTE para legibilidade) */
.flip-container.flipped .back {
    transform: rotateY(180deg);
}
.flip-container.flipped .back p, 
.flip-container.flipped .back hr,
.flip-container.flipped .back .st-emotion-cache-1l0353 {
    transform: rotateY(0deg); 
}


/* 3. Estilos de Conteúdo (CORES INVERTIDAS) */

/* Título principal (h1) AGORA É PRETO */
h1 { color: #1a1a1a; } 

/* Subtítulos (h2, h3) AGORA SÃO AZUL */
h2, h3 { color: #4255ff; }

/* Título da Palavra no Card (h2) AGORA É AZUL */
.flashcard-side h2 {
    color: #4255ff;
    font-size: 2.8em; 
    margin-bottom: 0.6em;
    text-transform: uppercase;
}
.flashcard-side p {
    color: #4a5568; /* Mantido cinza para corpo de texto padrão */
    font-size: 1.3em; 
    /* CORREÇÃO DE ESPAÇAMENTO DE LINHA (Menos espaçado) */
    line-height: 1.3; 
    margin-bottom: 5px; /* Reduz o espaço entre parágrafos padrão */
}


/* 4. Cores de Feedback */
.stSuccess {
    background-color: #f0f4ff;
    /* BORDA DE SUCESSO AGORA É PRETA */
    border-left: 5px solid #1a1a1a; 
}
.stError {
    background-color: #f7f9fa;
    border-left: 5px solid #8e949d;
}
</style>
""", unsafe_allow_html=True)


# --- 3. Carregamento de Recursos (Cachê) ---

@st.cache_resource
def load_nlp_resources():
    preencher_mascara = pipeline("fill-mask", model="distilbert-base-uncased")
    
    # Modelo para geração de texto (ainda útil para definições e paráfrases)
    gerador_texto = pipeline("text2text-generation", model="google/flan-t5-base") 
    
    # NOVO: MODELO DEDICADO PARA TRADUÇÃO EN -> PT (Helsinki-NLP/opus-mt-en-pt)
    # Baixa o modelo e o tokenizador OPUS-MT
    model_name = "Helsinki-NLP/opus-mt-en-pt"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # Cria o pipeline de tradução
    tradutor_en_pt = pipeline("translation_en_to_pt", model=model, tokenizer=tokenizer)
    
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    return preencher_mascara, gerador_texto, tradutor_en_pt, embedder

@st.cache_resource
def load_data():
    # Carrega dados locais ou usa dados mock se arquivos não existirem
    if not os.path.exists("10000_Words.csv") or not os.path.exists("templates.csv"):
        # Simulação de dados para teste
        palavras_base = ["word", "test", "example", "definition", "curiosity", "onomatopoeia", "floccinaucinihilipilification"]
        templates = ["The {palavra_alvo} is a good [MASK] of this rule.", "We should always [MASK] the {palavra_alvo} before using it."] 
        st.warning("Arquivos '10000_Words.csv' ou 'templates.csv' não encontrados. Usando dados de teste.")
        return palavras_base, templates
        
    df_palavras = pd.read_csv("10000_Words.csv", header=None)
    palavras_base = df_palavras[0].tolist()

    templates_df = pd.read_csv("templates.csv", skiprows=1, header=None)
    templates = templates_df[0].tolist()
    
    return palavras_base, templates


# --- 4. Inicializa Recursos e Funções ---
try:
    preencher_mascara, gerador_texto, tradutor_en_pt, embedder = load_nlp_resources()
except Exception as e:
    st.error(f"Falha ao carregar modelos de NLP. O aplicativo pode não funcionar corretamente. Erro: {e}")
    # Funções mock para evitar quebras
    def preencher_mascara(*args, **kwargs):
        return [{"sequence": "This is a [MASK] example sentence.", "score": 1.0}]
    def gerador_texto(*args, **kwargs):
        return [{"generated_text": "A simulated definition."}]
    def tradutor_en_pt(text):
         return [{"translation_text": f"[ERRO_MODELO: {text}]"}]
    
    class MockEmbedder:
        def encode(self, text): return [0]
    embedder = MockEmbedder()

palavras_base, templates = load_data()

# --- NOVO: Função de Tradução (usando o OPUS-MT dedicado) ---
def translate_text(text):
    """Traduz o texto usando o modelo OPUS-MT EN->PT."""
    # O modelo OPUS-MT é mais rápido e preciso para este par de idiomas.
    try:
        # O pipeline já sabe a direção (en_to_pt)
        resp = tradutor_en_pt(text, max_length=128, do_sample=False)[0]["translation_text"].strip()
        # Limpeza para remover artefatos de tradução
        return resp.replace('<pad>', '').strip()
    except Exception:
        return f"[Erro de Tradução: {text}]"

# --- As Funções do Flashcard (Lógica) ---

def generate_example(word):
    template = random.choice(templates)
    
    if "[MASK]" not in template:
        return {"sentence": "Template de exemplo inválido.", "score": 0.0, "template": template}

    sentence_with_mask = template.replace("{palavra_alvo}", word)

    with st.spinner('Gerando exemplo de uso...'):
        try:
            # Usa o pipeline de preencher máscara
            result = preencher_mascara(sentence_with_mask, top_k=1)[0]
            sentence = result["sequence"].replace("[MASK]", word) 
            return {
                "sentence": sentence, 
                "score": result["score"],
                "template": template
            }
        except Exception:
            return {"sentence": f"Erro na geração de exemplo para '{word}'.", "score": 0.0, "template": template}

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

    # Recorre a Flan-T5 (agora gerador_texto)
    prompt = (
        f"Give a concise, dictionary-style definition of the English word '{word}'. "
        "5–15 words. No examples, no comparisons."
    )
    with st.spinner('Gerando definição com Flan-T5...'):
        try:
            resp = gerador_texto(prompt, max_length=32, do_sample=False)[0]["generated_text"].strip()
            return resp
        except Exception:
            return f"No definition found for {word}."

def generate_synonyms_antonyms(word, top_n=5, similarity_threshold=0.55):
    # Lógica de Sinônimos/Antônimos (depende do embedder e WordNet)
    try:
        with st.spinner('Buscando sinônimos e antônimos...'):
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
        # Usa Flan-T5 para paráfrase
        resp = gerador_texto(prompt, max_length=80, do_sample=False)[0]["generated_text"].strip()
        if resp and resp.lower() != definition.lower():
            return resp
        return f"It refers to {definition}"
    except Exception:
        return f"It refers to {definition}"

def generate_quiz(word, definition):
    question = f"What does **'{word}'** mean?"
    correct = rephrase_definition(definition, word)
    
    # Pool de respostas falsas (mantido extenso para garantir diversidade)
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
        "A cave filled com crystals that sang.", "A forgotten bridge de light.", 
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
        # Garante que a resposta correta está na lista, substituindo um dos fakes
        options[random.randint(0, 3)] = correct 

    return {"question": question, "options": options, "answer": correct}

def gerar_flashcard(word):
    with st.empty(): 
        with st.spinner(f"Gerando Flashcard para '{word}'..."):
            exemplo = generate_example(word)
            definicao = generate_definition(word)
            
            # --- USO DO NOVO MODELO DE TRADUÇÃO ---
            traducao_definicao = translate_text(definicao)
            traducao_exemplo = translate_text(exemplo["sentence"])
            # ------------------------------------
            
            synonyms, antonyms = generate_synonyms_antonyms(word)
            quiz = generate_quiz(word, definicao)
            
            return {
                "word": word,
                "example_sentence": exemplo["sentence"],
                "definition": definicao,
                "translation_definition": traducao_definicao, 
                "translation_example": traducao_exemplo,     
                "synonyms": synonyms,
                "antonyms": antonyms,
                "quiz": quiz
            }

# --- 5. Lógica da Interface Streamlit (Estilo Quizlet) ---

st.title(" 🧠 Review.IA – Flashcards ")

if "card_flipped" not in st.session_state:
    st.session_state["card_flipped"] = False

# FUNÇÃO: Flip Card e Rerun
def flip_card():
    st.session_state["card_flipped"] = not st.session_state["card_flipped"]
    st.rerun() 

# --- SIDEBAR: Input e Controles ---
with st.sidebar:
    st.header("⚙️ Controles")
    palavra_padrao = st.session_state.get("flashcard", {}).get("word", "curiosity")
    palavra = st.text_input("Digite uma palavra em inglês:", key="word_input_sb", value=palavra_padrao)
    
    if st.button("Gerar Flashcard", use_container_width=True, type="primary") and palavra:
        if palavra.strip():
            st.session_state["flashcard"] = gerar_flashcard(palavra.strip())
            st.session_state["quiz_answered"] = False
            st.session_state["quiz_feedback"] = ""
            st.session_state["card_flipped"] = False 
            st.rerun()


# --- CORPO PRINCIPAL: Card Interativo ---
if "flashcard" in st.session_state:
    flashcard = st.session_state["flashcard"]
    
    # 1. CARD PRINCIPAL (Clique para Virar)
    st.subheader(f"🎴 Flashcard: {flashcard['word'].upper()}")
    
    flip_class = "flipped" if st.session_state["card_flipped"] else ""
    
    # Decodifica HTML para garantir que o texto seja renderizado corretamente
    decoded_definition = html.unescape(flashcard['definition'])
    decoded_example = html.unescape(flashcard['example_sentence'])
    decoded_translation_definition = html.unescape(flashcard['translation_definition'])
    decoded_translation_example = html.unescape(flashcard['translation_example'])

    card_html_parts = [
        f'<div class="flip-container {flip_class}">',
        '    <div class="flipper">',
        # FRENTE DO CARD
        '        <div class="front flashcard-side">',
        f'            <h2>{flashcard["word"].upper()}</h2>', 
        f'            <p style="color:#1a1a1a; font-weight:bold; margin-bottom: 0;">Clique no botão abaixo para ver a definição.</p>',
        '        </div>',
        # VERSO DO CARD (CORES INVERTIDAS)
        '        <div class="back flashcard-side">',
        f'            <p style="color:#1a1a1a; font-size: 1.5em; font-weight:bold; margin-bottom: 5px;">📚 Definição (EN)</p>',
        f'            <p style="color:#4255ff; font-size: 1.3em; line-height: 1.3; margin-bottom: 5px;">{decoded_definition}</p>',
        
        # --- TRADUÇÃO DA DEFINIÇÃO ---
        f'            <p style="color:#1a1a1a; font-size: 1.0em; font-weight:normal; line-height: 1.3; margin-bottom: 20px;">Tradução (PT): {decoded_translation_definition}</p>',
        # ------------------------------------

        f'            <hr style="border-top: 1px solid #1a1a1a; width: 50%;">', 
        
        # --- EXEMPLO E TRADUÇÃO DO EXEMPLO ---
        f'            <p style="font-style: italic; color: #4a5568; font-size: 1.2em; line-height: 1.3; margin-bottom: 5px;">📝 Exemplo (EN): {decoded_example}</p>',
        f'            <p style="font-style: italic; color: #1a1a1a; font-size: 0.9em; line-height: 1.3; margin-bottom: 0;">Tradução (PT): {decoded_translation_example}</p>',
        # ---------------------------------------------
        
        '        </div>',
        '    </div>',
        '</div>'
    ]
    card_html = "\n".join(card_html_parts)

    # Renderiza a estrutura HTML
    st.markdown(card_html, unsafe_allow_html=True)
    
    # Botão Streamlit que aciona a função flip_card
    if st.button("Virar Card", key="flip_button", use_container_width=True):
        flip_card()
    
    
    st.markdown("---")


    # 2. DETALHES E RELACIONADOS
    st.subheader("💡 Informações Relacionadas")
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"**🟩 Sinônimos:** {', '.join(flashcard['synonyms']) if flashcard['synonyms'] else 'Nenhum encontrado.'}")
    with col2:
        st.error(f"**🟥 Antônimos:** {', '.join(flashcard['antonyms']) if flashcard['antonyms'] else 'Nenhum encontrado.'}")
    
    st.markdown("---")


    # 3. QUIZ DE MÚLTIPLA ESCOLHA
    st.subheader("📝 Teste seu Conhecimento")
    quiz = flashcard["quiz"]
    
    st.markdown(f"**❓ Pergunta:** {quiz['question']}")
    
    if not st.session_state.get("quiz_answered", False):
        
        with st.form(key='quiz_form'):
            selected_option = st.radio("Escolha a opção correta:", quiz["options"], key="radio_quiz_options")
            submit_button = st.form_submit_button(label='Responder')

            if submit_button:
                resposta_escolhida = selected_option
                if resposta_escolhida == quiz["answer"]:
                    st.session_state["quiz_feedback"] = "🎉 **Correto!** Excelente trabalho."
                    st.session_state["quiz_answered"] = True
                else:
                    st.session_state["quiz_feedback"] = f"❌ **Errado.** A resposta certa era: **{quiz['answer']}**"
                    st.session_state["quiz_answered"] = True
                st.rerun()

    # Bloco de Feedback e Botões de Ação
    if st.session_state.get("quiz_answered", False):
        st.markdown(st.session_state["quiz_feedback"])
        
        col_quiz_novo, col_limpar = st.columns(2)
        
        with col_quiz_novo:
            if st.button("🔄 Gerar Novo Quiz", use_container_width=True):
                nova_quiz = generate_quiz(flashcard["word"], flashcard["definition"])
                st.session_state["flashcard"]["quiz"] = nova_quiz
                st.session_state["quiz_answered"] = False
                st.session_state["quiz_feedback"] = ""
                st.rerun()

        with col_limpar:
            if st.button("🗑️ Limpar e Tentar Outra Palavra", use_container_width=True, type="primary"): 
                st.session_state.pop("flashcard", None)
                st.session_state["quiz_answered"] = False
                st.session_state["quiz_feedback"] = ""
                st.session_state["card_flipped"] = False
                st.rerun()

        st.markdown("---")
        st.info("Para gerar um flashcard de uma palavra diferente, use a barra lateral **⚙️ Controles**.")