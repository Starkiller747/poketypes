# 🧬 Pokémon Type Predictor

A Streamlit web app that predicts the **Pokémon type(s)** based on the name input using a deep learning model trained on real Pokédex data.

---

## 🚀 Features

- Predicts one or two types for a given Pokémon name
- Uses a character-level neural network trained on the full Pokédex
- Confidence chart for predicted type probabilities
- Cleaned data from [PokémonDB](https://pokemondb.net/pokedex/all)
- Optional: Visual display of Pokémon type icons (if available)

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **TensorFlow / Keras** (Neural network training)
- **Scikit-learn** (Naive Bayes, preprocessing)
- **Streamlit** (Web app UI)
- **BeautifulSoup + Requests** (Web scraping Pokédex data)
- **Joblib** (Model and vectorizer serialization)
- **Altair** (Interactive bar chart visualization)

---

## 📦 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/pokemon-type-predictor.git
cd pokemon-type-predictor
