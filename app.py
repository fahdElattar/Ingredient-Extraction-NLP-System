from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
import nltk
import string
import re
import os

app = Flask(__name__)

# --- CONFIGURATION ---
MAX_LEN = 22

# --- LOAD MODELS LOCALLY ---
model = load_model('models/ner_model.h5')
w2v_model = Word2Vec.load('models/word2vec.model')
label_encoder = joblib.load('models/label_encoder.pkl')

nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path.append(nltk_data_path)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stop_words.discard('and')

# Define punctuation to remove
punct_to_remove = set(string.punctuation) - {',', '/', '\\'}
punct_to_remove.update(['-lrb-', '-rrb-'])

def vectorize_tokens(tokens, model):
    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
        else:
            vectors.append(np.zeros(model.vector_size))
    return np.array(vectors)

def clean_and_align(tokens_str, labels_str):
    tokens = tokens_str.split()
    labels = labels_str.split(', ')
    cleaned_tokens, cleaned_labels = [], []

    for tok, lab in zip(tokens, labels):
        tok_lower = tok.lower().strip()
        if tok_lower in punct_to_remove:
            continue
        if re.match(r'^[a-z0-9/]+-[a-z]+$', tok_lower):
            parts = tok_lower.split('-')
            for p in parts:
                cleaned_tokens.append(lemmatizer.lemmatize(p))
                cleaned_labels.append(lab)
            continue
        if tok_lower in stop_words and tok_lower != "and":
            continue
        cleaned_tokens.append(lemmatizer.lemmatize(tok_lower))
        cleaned_labels.append(lab)
    return " ".join(cleaned_tokens), ", ".join(cleaned_labels)

def convert_to_iob(labels):
    iob_labels = []
    prev_label = "O"
    for label in labels:
        if label == "O":
            iob_labels.append("O")
            prev_label = "O"
        else:
            if label != prev_label:
                iob_labels.append("B-" + label)
            else:
                iob_labels.append("I-" + label)
            prev_label = label
    return iob_labels

def fix_iob_sequence(labels):
    """
    Ensures every I-XXX follows a valid B-XXX or I-XXX of the same type.
    If not, converts it to B-XXX.
    """
    fixed = []
    prev_label = 'O'
    changes = 0

    for lbl in labels:
        if lbl.startswith('I-'):
            curr_type = lbl.split('-')[1]
            prev_type = prev_label.split('-')[1] if '-' in prev_label else None

            # Invalid case: previous not of same entity or not B/I
            if not (prev_label.startswith(('B-', 'I-')) and prev_type == curr_type):
                lbl = f'B-{curr_type}'
                changes += 1
        fixed.append(lbl)
        prev_label = lbl

    return fixed, changes

def predict_ingredient_labels(sentences):
    """
    Predict IOB labels for a list of ingredient sentences using trained model.
    Returns list of token-label pairs per sentence, filtering out 'O' labels.
    """

    # Step 1: Create DataFrame like test.csv, one row per sentence
    df = pd.DataFrame({
        "source": ["user"]*len(sentences),
        "ingredient_id": list(range(len(sentences))),
        "tokens_joined": [" ".join(sent.split()) for sent in sentences],  # space-separated tokens
        "labels_joined": [", ".join(['O']*len(sent.split())) for sent in sentences]  # dummy 'O' labels
    })

    # Step 2: Clean and align tokens
    df[['tokens_cleaned', 'labels_cleaned']] = df.apply(
        lambda row: pd.Series(clean_and_align(row['tokens_joined'], row['labels_joined'])),
        axis=1
    )

    # Step 3: Tokenize
    df['tokens'] = df['tokens_cleaned'].apply(lambda x: x.split())
    df['labels'] = df['labels_cleaned'].apply(lambda x: x.split(', '))

    # Step 4: Convert to IOB and fix
    df['iob_labels'] = df['labels'].apply(convert_to_iob)
    df['iob_labels'] = df['iob_labels'].apply(lambda x: fix_iob_sequence(x)[0])

    # Step 5: Vectorize tokens
    df['word_vectors'] = df['tokens'].apply(lambda x: vectorize_tokens(x, w2v_model))

    # Step 6: Pad sequences
    X_padded = pad_sequences(
        df['word_vectors'].tolist(),
        maxlen=MAX_LEN,
        dtype='float32',
        padding='post',
        truncating='post',
        value=0.0
    )

    # Step 7: Predict
    y_pred_probs = model.predict(X_padded)
    y_pred = np.argmax(y_pred_probs, axis=-1)

    # Step 8: Decode predictions and filter out 'O'
    predictions = []
    for i, tokens in enumerate(df['tokens']):
        pred_seq = []
        for j, token in enumerate(tokens):
            if j < MAX_LEN:
                label = label_encoder.inverse_transform([y_pred[i][j]])[0]
                if label != 'O':
                    clean_label = str(label).replace('B-', '').replace('I-', '')
                    pred_seq.append((str(token), clean_label))
        
        predictions.append({'entities': pred_seq})

    return predictions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    recipe_text = request.form.get('recipe', '').strip()
    if not recipe_text:
        return render_template('index.html', error="Empty recipe!")

    sentences = [line.strip() for line in recipe_text.split('\n') if line.strip()]
    extracted = predict_ingredient_labels(sentences)
    return render_template('index.html', recipe=recipe_text, ingredients=extracted)

if __name__ == '__main__':
    app.run(port=5000, debug=True)