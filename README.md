# 🧠 Review.IA – Flashcards Quizlet Style

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Natural%20Language%20Processing-00BFFF?style=for-the-badge)
![AI](https://img.shields.io/badge/AI-8A2BE2?style=for-the-badge)
![Flashcards](https://img.shields.io/badge/Flashcards-FF4500?style=for-the-badge)
![Quiz](https://img.shields.io/badge/Quiz-D2691E?style=for-the-badge)

---

## 🌟 Project Overview

**Review.IA** is an interactive web application built with **Streamlit** that simulates a Quizlet-style flashcard learning experience, enhanced with **Artificial Intelligence**.  

The main goal is to **dynamically generate study materials** for any English word provided, including:  
- Definition of the word  
- Example usage  
- Multiple-choice quiz to test the user  

All in **real-time** and fully interactive!  

---

## 🎴 Demo

![Flashcard Flip Demo](https://media.giphy.com/media/3o6Zt8MgUuvSbkZYWc/giphy.gif)  
*Interactive 3D flashcard flipping effect*

---

## ✨ Key Features

- **🎴 Interactive 3D Flashcards:** Click to flip, simulating physical flashcards  
- **🤖 AI & NLP Generated Content:** Uses **WordNet** and NLP models to generate definitions, examples, synonyms, and antonyms  
- **📝 Dynamic Quiz:** Multiple-choice questions with instant feedback  
- **🔗 Semantic Filtering:** Uses `sentence-transformers` to select synonyms by semantic similarity  
- **🎨 User-Friendly Interface:** Streamlit app with custom CSS styling for a clean learning experience  

---

## 🛠️ Technologies Used

| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **Web Framework** | `Streamlit` | Interactive web interface |
| **NLP Core** | `NLTK (WordNet)` | Definitions, synonyms, antonyms |
| **AI Models** | `Hugging Face Transformers` | `distilbert-base-uncased` for masked examples, `google/flan-t5-base` for definitions/paraphrasing |
| **Semantic Similarity** | `Sentence-Transformers` | Embeddings for filtering synonyms/antonyms |
| **Data Handling** | `Pandas` | Load and manage word lists & templates |
| **Styling** | `HTML/CSS` | Custom 3D flashcards and UI design |

---

## 🚀 How to Run Locally

### 1️⃣ Prerequisites
- Python 3.8+ installed

### 2️⃣ Clone the Repository

git clone https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
cd SEU_REPOSITORIO
3️⃣ Create & Activate Virtual Environment
# Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate

4️⃣ Install Dependencies
pip install streamlit nltk pandas transformers sentence-transformers


(or use pip install -r requirements.txt)

5️⃣ Prepare Data Files

10000_Words.csv → List of words

templates.csv → Example sentence templates

(If missing, the app will use mock data for testing)

6️⃣ Run the App
streamlit run SEU_ARQUIVO_PRINCIPAL.py


Open your browser at http://localhost:8501 to explore the app.

🤝 Contributing

Contributions are welcome!

Open an Issue for suggestions or bug reports

Send a Pull Request for improvements

📂 Project Files

SEU_ARQUIVO_PRINCIPAL.py
 – Main Streamlit app

10000_Words.csv
 – Word list for flashcards

templates.csv
 – Sentence templates for examples
