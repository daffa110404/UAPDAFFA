import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import re
import joblib
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ==========================================
# 1. SETUP HALAMAN & JUDUL
# ==========================================
st.set_page_config(page_title="Deteksi Komentar Toxic", page_icon="🚫")

st.title("🚫 Deteksi Komentar Toxic")
st.write("Aplikasi ini menggunakan AI untuk mendeteksi apakah sebuah komentar bersifat **Toxic** atau **Aman**.")

# ==========================================
# 2. DEFINISI KELAS LSTM (Wajib sama persis dengan saat training)
# ==========================================
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        super(SentimentLSTM, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        sig_out = self.sigmoid(out)
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]
        return sig_out, hidden

    def init_hidden(self, batch_size):
        device = torch.device('cpu') # Kita paksa CPU untuk website biar aman
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden

# ==========================================
# 3. FUNGSI LOAD MODEL (Di-Cache biar cepat)
# ==========================================
@st.cache_resource
def load_lstm_resources():
    # Load Vocab
    vocab = joblib.load('models/vocab.pkl')
    # Setup Model
    vocab_size = len(vocab) + 1
    model = SentimentLSTM(vocab_size, 1, 200, 128, 2)
    # Load Weight
    model.load_state_dict(torch.load('models/model_lstm.pth', map_location=torch.device('cpu')))
    model.eval()
    return model, vocab

@st.cache_resource
def load_indobert():
    path = 'models/model_bert/'
    tokenizer = BertTokenizer.from_pretrained(path)
    model = BertForSequenceClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_distilbert():
    path = 'models/model_distilbert/'
    tokenizer = DistilBertTokenizer.from_pretrained(path)
    model = DistilBertForSequenceClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model

# ==========================================
# 4. FUNGSI PREPROCESSING & PREDIKSI
# ==========================================
stop_factory = StopWordRemoverFactory()
stopword = stop_factory.create_stop_word_remover()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+','', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = stopword.remove(text)
    return text

def predict_lstm(text, model, vocab):
    # Preprocess
    text = clean_text(text)
    words = text.split()
    encoded = [vocab.get(w, 0) for w in words]
    # Padding
    seq_length = 100
    features = np.zeros((1, seq_length), dtype=int)
    features[0, -len(encoded):] = np.array(encoded)[:seq_length]
    # Predict
    input_tensor = torch.from_numpy(features)
    h = model.init_hidden(1)
    output, _ = model(input_tensor, h)
    pred = output.item()
    return pred # Nilai 0.0 - 1.0

def predict_bert(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs[0][1].item() # Ambil probabilitas kelas Toxic (label 1)

# ==========================================
# 5. UI UTAMA (Tampilan User)
# ==========================================
col1, col2 = st.columns([3, 1])

with col1:
    input_text = st.text_area("Masukkan Komentar:", height=150, placeholder="Contoh: Dasar kamu bodoh banget tidak berguna!")

with col2:
    st.write("Pilih Model:")
    model_option = st.radio("Model AI", ["IndoBERT (Terbaik)", "DistilBERT (Cepat)", "LSTM (Basic)"])

if st.button("🔍 Analisis Komentar"):
    if input_text:
        with st.spinner('Sedang menganalisis...'):
            try:
                # Logika pemilihan model
                score = 0
                if "LSTM" in model_option:
                    model, vocab = load_lstm_resources()
                    score = predict_lstm(input_text, model, vocab)
                elif "IndoBERT" in model_option:
                    tokenizer, model = load_indobert()
                    score = predict_bert(input_text, tokenizer, model)
                elif "DistilBERT" in model_option:
                    tokenizer, model = load_distilbert()
                    score = predict_bert(input_text, tokenizer, model)

                # Tampilkan Hasil
                st.markdown("---")
                if score > 0.5:
                    st.error(f"⚠️ **TOXIC DETECTED!**")
                    st.write(f"Tingkat Keyakinan: **{score*100:.2f}%**")
                    st.progress(int(score*100))
                else:
                    st.success(f"✅ **KOMENTAR AMAN**")
                    st.write(f"Tingkat Keyakinan Toxic: **{score*100:.2f}%** (Rendah)")
                    st.progress(int(score*100))
                    
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
                st.warning("Pastikan folder 'models' sejajar dengan file app.py")
    else:
        st.warning("Mohon isi kolom komentar terlebih dahulu.")