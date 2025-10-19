import streamlit as st

st.set_page_config(page_title="Home- Review.IA")

st.title("🧠Home- Review.IA – Flashcards")
st.markdown("---")
st.info("Aqui você pode acessar as páginas do app.")

# Opcional: Adicionar um botão de volta
if st.button("⬅️ Abrir Review.IA- English"):
    # O arquivo principal SEMPRE deve ser referenciado pelo seu nome (ex: app_principal.py)
    st.switch_page("Pages/English.py")

if st.button("⬅️ Abrir Review.IA- PT-br"):
    # O arquivo principal SEMPRE deve ser referenciado pelo seu nome (ex: app_principal.py)
    st.switch_page("Pages/PT-br.py")