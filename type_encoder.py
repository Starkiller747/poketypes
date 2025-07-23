import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib

# Global variables for training
mlb = MultiLabelBinarizer()
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
model = None

def save_artifacts(model, vectorizer, mlb):
    joblib.dump(model, "model.joblib")
    joblib.dump(vectorizer, "vectorizer.joblib")
    joblib.dump(mlb, "mlb.joblib")

def fetch_pokemon_data():
    url = 'https://pokemondb.net/pokedex/all'
    req = requests.get(url)
    if req.status_code != 200:
        raise Exception("Failed to retrieve data")

    soup = BeautifulSoup(req.text, 'html.parser')
    names = [name.get_text() for name in soup.find_all(class_='cell-name')]
    types = [icon.get_text() for icon in soup.find_all(class_='cell-icon')]

    df = pd.DataFrame(list(zip(names, types)), columns=['Name', 'Type1'])

    # Clean data
    df['Name'] = df['Name'].str.replace(
        r'^(\w+)(?:\s.*)?\b(Form|Forme|Mask|Plumage|Style|Size|Mode|Rotom|Cloak|Face|Rider|Necrozma|Rockruff|Confined|Unbound|Hero of Many Battles|Crowned Sword|Crowned Shield)\b',
        r'\1',
        regex=True
    )

    keywords = ['Hisuian', 'Galarian', 'Paldean', 'Breed', 'Partner', 'Mega', 'Ash-', 'Bloodmoon', 'Eternamax']
    df = df[~df['Name'].str.contains('|'.join(keywords), case=True, na=False)].copy()

    df['Name'] = df['Name'].str.replace(r'\b(Male|Female|)\b', '', regex=True)
    df['Name'] = df['Name'].str.replace(r'\bFamily of \w+\b', '', regex=True)
    df['Name'] = df['Name'].str.strip()
    df.drop_duplicates(keep='first', inplace=True)
    df[['Type1', 'Type2']] = df['Type1'].str.split(' ', n=1, expand=True)
    df.reset_index(drop=True, inplace=True)

    df["Types"] = df[["Type1", "Type2"]].apply(lambda x: [t for t in x if pd.notna(t) and t != ''], axis=1)

    return df

def prepare_data(df):
    y = pd.DataFrame(mlb.fit_transform(df["Types"]), columns=mlb.classes_)
    y = y.loc[:, y.columns != '']
    X = vectorizer.fit_transform(df["Name"])
    return train_test_split(X, y, test_size=0.2, random_state=77)

def train_dense_model(X_train, y_train, X_val, y_val):
    global model
    model = Sequential([
        Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(y_train.shape[1], activation='sigmoid')
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(X_train.toarray(), y_train, epochs=100, batch_size=8, validation_data=(X_val.toarray(), y_val))

    save_artifacts(model, vectorizer, mlb)
    return model

def predict_type(name, model, vectorizer, mlb):
    name_vector = vectorizer.transform([name])
    prediction = model.predict(name_vector.toarray())

    i = 5
    while True:
        prediction_binary = (prediction > i / 100).astype(int)
        if np.any(prediction_binary == 1) or i <= 0:
            break
        i -= 0.01

    if not np.any(prediction_binary == 1):
        return ["Normal"], "Confidence %: 0.00", {}

    type_confidences = {
    mlb.classes_[idx]: float(prediction[0][idx])
    for idx in range(len(prediction[0]))}

    sorted_types = sorted(type_confidences.items(), key=lambda x: x[1], reverse=True)

    prediction_binary = (prediction > i / 100).astype(int)
    selected_types = [(t, s) for t, s in sorted_types
        if prediction_binary[0][mlb.classes_.tolist().index(t)] == 1]
    
    top_types = [t[0] for t in selected_types[:2]]
    top_types_confidences = dict(selected_types[:5])

    return top_types, top_types_confidences

def main():
    df = fetch_pokemon_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    train_dense_model(X_train, y_train, X_test, y_test)
    #types, conf, scores = predict_type("John", model, vectorizer, mlb)
    #print(types, conf, scores)

if __name__ == "__main__":
    main()
