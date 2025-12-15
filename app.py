import os
import base64
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from flask import Flask, request, render_template_string

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

app = Flask(__name__)

DATASET_PATH = "diabetes.csv"
RANDOM_STATE = 42

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET = "Outcome"

# ---- Global cache biar gak training berulang-ulang tiap request ----
MODEL = None
SCALER = None
BEST_K = None
METRICS_TEXT = None
CM = None
PLOT_B64 = None


def train_model():
    """
    Alur sesuai modul:
    Load -> Split -> Scale -> Cari K (1..40) -> Evaluasi -> Simpan model terbaik
    """
    global MODEL, SCALER, BEST_K, METRICS_TEXT, CM, PLOT_B64

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"File {DATASET_PATH} tidak ditemukan. Pastikan diabetes.csv ada di folder yang sama dengan app.py"
        )

    df = pd.read_csv(DATASET_PATH)

    # Basic check
    for col in FEATURES + [TARGET]:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ada di dataset. Cek header diabetes.csv")

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # 1) Split dulu (PENTING: jangan scaling sebelum split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # 2) Scaling WAJIB untuk KNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # fit hanya di train
    X_test_scaled = scaler.transform(X_test)         # test hanya transform

    # 3) Cari K terbaik (1..40) pakai error rate di test
    ks = list(range(1, 41))
    error_rates = []
    best_k = None
    best_acc = -1.0
    best_model = None

    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        err = 1.0 - acc

        error_rates.append(err)

        if acc > best_acc:
            best_acc = acc
            best_k = k
            best_model = knn

    # 4) Evaluasi model final (pakai K terbaik)
    y_pred_final = best_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred_final)
    report = classification_report(y_test, y_pred_final, digits=4)

    # 5) Plot error rate vs K (Elbow Method)
    fig = plt.figure()
    plt.plot(ks, error_rates)
    plt.xlabel("Nilai K")
    plt.ylabel("Error Rate (1 - Accuracy)")
    plt.title("Elbow Method untuk mencari K terbaik")

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    plot_b64 = base64.b64encode(buf.read()).decode("utf-8")

    # Simpan ke global
    MODEL = best_model
    SCALER = scaler
    BEST_K = best_k
    CM = cm
    METRICS_TEXT = report
    PLOT_B64 = plot_b64


def ensure_trained():
    global MODEL
    if MODEL is None:
        train_model()


HTML = """
<!doctype html>
<html lang="id">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TA-10 KNN Diabetes Prediction</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    
    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh;
      padding: 20px;
      color: #2d3748;
      line-height: 1.6;
    }
    
    .container {
      max-width: 1200px;
      margin: 0 auto;
    }
    
    .header {
      text-align: center;
      color: white;
      margin-bottom: 40px;
      padding: 30px 20px;
    }
    
    .header h1 {
      font-size: 2.5rem;
      font-weight: 700;
      margin-bottom: 10px;
      text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    
    .header .subtitle {
      font-size: 1.1rem;
      opacity: 0.95;
      font-weight: 300;
    }
    
    .card {
      background: white;
      border-radius: 20px;
      padding: 30px;
      margin-bottom: 30px;
      box-shadow: 0 10px 40px rgba(0,0,0,0.1);
      transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card:hover {
      transform: translateY(-5px);
      box-shadow: 0 15px 50px rgba(0,0,0,0.15);
    }
    
    .card h2 {
      font-size: 1.8rem;
      font-weight: 600;
      color: #667eea;
      margin-bottom: 20px;
      padding-bottom: 15px;
      border-bottom: 3px solid #e2e8f0;
    }
    
    .card h3 {
      font-size: 1.4rem;
      font-weight: 600;
      color: #4a5568;
      margin-top: 25px;
      margin-bottom: 15px;
    }
    
    .card p {
      color: #718096;
      margin-bottom: 15px;
    }
    
    label {
      display: block;
      margin-top: 15px;
      margin-bottom: 8px;
      font-weight: 500;
      color: #4a5568;
      font-size: 0.95rem;
    }
    
    input[type="number"] {
      width: 100%;
      padding: 12px 16px;
      border-radius: 10px;
      border: 2px solid #e2e8f0;
      font-size: 1rem;
      transition: all 0.3s ease;
      background: #f7fafc;
    }
    
    input[type="number"]:focus {
      outline: none;
      border-color: #667eea;
      background: white;
      box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    .row {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 20px;
    }
    
    @media (max-width: 768px) {
      .row {
        grid-template-columns: 1fr;
      }
      .header h1 {
        font-size: 2rem;
      }
      .card {
        padding: 20px;
      }
    }
    
    button {
      margin-top: 25px;
      padding: 14px 32px;
      border-radius: 12px;
      border: none;
      cursor: pointer;
      font-size: 1rem;
      font-weight: 600;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      transition: all 0.3s ease;
      box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    button:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    button:active {
      transform: translateY(0);
    }
    
    .result-box {
      margin-top: 25px;
      padding: 20px;
      border-radius: 12px;
      font-size: 1.1rem;
      animation: slideIn 0.5s ease;
    }
    
    @keyframes slideIn {
      from {
        opacity: 0;
        transform: translateY(-10px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }
    
    .ok {
      background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
      color: white;
      border: none;
    }
    
    .bad {
      background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
      color: white;
      border: none;
    }
    
    .result-box b {
      font-size: 1.2rem;
    }
    
    .probability {
      margin-top: 15px;
      padding: 12px;
      background: rgba(255,255,255,0.2);
      border-radius: 8px;
      font-size: 1rem;
    }
    
    pre {
      white-space: pre-wrap;
      background: #f7fafc;
      padding: 20px;
      border-radius: 10px;
      border: 1px solid #e2e8f0;
      overflow-x: auto;
      font-size: 0.9rem;
      line-height: 1.5;
      color: #2d3748;
    }
    
    .info-box {
      background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
      color: white;
      padding: 15px 20px;
      border-radius: 12px;
      margin-bottom: 25px;
      font-size: 1.1rem;
    }
    
    .info-box .k-value {
      font-size: 1.3rem;
      font-weight: 700;
      margin-top: 5px;
    }
    
    img {
      max-width: 100%;
      border-radius: 15px;
      box-shadow: 0 5px 20px rgba(0,0,0,0.1);
      margin: 20px 0;
    }
    
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 15px;
      margin-top: 20px;
    }
    
    .stat-card {
      background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
      padding: 20px;
      border-radius: 12px;
      text-align: center;
      border: 2px solid #e2e8f0;
    }
    
    .stat-card .stat-label {
      font-size: 0.9rem;
      color: #718096;
      margin-bottom: 8px;
      font-weight: 500;
    }
    
    .stat-card .stat-value {
      font-size: 1.8rem;
      font-weight: 700;
      color: #667eea;
    }
    
    ul {
      list-style: none;
      padding-left: 0;
    }
    
    ul li {
      padding: 12px 0;
      padding-left: 30px;
      position: relative;
      color: #4a5568;
    }
    
    ul li:before {
      content: "✓";
      position: absolute;
      left: 0;
      color: #48bb78;
      font-weight: bold;
      font-size: 1.2rem;
    }
    
    .badge {
      display: inline-block;
      padding: 6px 12px;
      background: #667eea;
      color: white;
      border-radius: 20px;
      font-size: 0.85rem;
      font-weight: 500;
      margin-left: 10px;
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>🔬 TA-10: KNN Diabetes Prediction</h1>
      <p class="subtitle">Machine Learning untuk Klasifikasi Diabetes menggunakan K-Nearest Neighbors</p>
    </div>

    <div class="card">
      <h2>📊 1. Prediksi Data Baru</h2>
      <p style="color: #718096; margin-bottom: 25px;">Masukkan data pasien untuk melakukan prediksi diabetes menggunakan model KNN yang telah dilatih.</p>
      
      <form method="POST" action="/predict">
        <div class="row">
          {% for f in features %}
            <div>
              <label for="{{ f }}">{{ f }}</label>
              <input 
                id="{{ f }}"
                name="{{ f }}" 
                type="number" 
                step="any" 
                required 
                value="{{ request.form.get(f,'') }}"
                placeholder="Masukkan nilai {{ f }}"
              >
            </div>
          {% endfor %}
        </div>
        <button type="submit">🚀 Prediksi Sekarang</button>
      </form>

      {% if pred is not none %}
        {% if pred == 1 %}
          <div class="result-box bad">
            <b>⚠️ Hasil Prediksi: DIABETES (1)</b>
            <div class="probability">
              <b>Probabilitas:</b> {{ prob }}{% if prob_percent is not none %} ({{ prob_percent }}%){% endif %}
            </div>
          </div>
        {% else %}
          <div class="result-box ok">
            <b>✓ Hasil Prediksi: TIDAK DIABETES (0)</b>
            <div class="probability">
              <b>Probabilitas:</b> {{ prob }}{% if prob_percent is not none %} ({{ prob_percent }}%){% endif %}
            </div>
          </div>
        {% endif %}
      {% endif %}
    </div>

    <div class="card">
      <h2>🎯 2. Training & Evaluasi Model</h2>
      <p style="color: #718096;">Alur pelatihan: Split Data → Scaling → Pencarian K Optimal (1-40) → Evaluasi → Prediksi</p>
      
      <form method="GET" action="/train">
        <button type="submit">⚙️ Jalankan Training & Evaluasi</button>
      </form>

      {% if best_k %}
        <div class="info-box">
          <div>✨ K Terbaik yang Ditemukan:</div>
          <div class="k-value">{{ best_k }}</div>
        </div>

        <h3>📈 Grafik Error Rate vs K (Elbow Method)</h3>
        <p style="color: #718096; margin-bottom: 15px;">Grafik ini menunjukkan hubungan antara nilai K dan error rate untuk menemukan nilai K optimal.</p>
        <img src="data:image/png;base64,{{ plot_b64 }}" alt="Elbow Method Chart" />

        <h3>📋 Confusion Matrix</h3>
        <pre>{{ cm }}</pre>

        <h3>📊 Classification Report</h3>
        <pre>{{ report }}</pre>
      {% endif %}
    </div>

    <div class="card">
      <h2>ℹ️ Catatan Penting</h2>
      <ul>
        <li>Pastikan file <b>diabetes.csv</b> tersedia di direktori yang sama dengan aplikasi saat deploy ke Railway</li>
        <li>Jika terjadi error, periksa nama kolom dataset dan pastikan sesuai dengan header yang digunakan</li>
        <li>Model menggunakan StandardScaler untuk normalisasi data sebelum prediksi</li>
        <li>Pencarian K optimal dilakukan dengan rentang nilai 1 hingga 40 menggunakan test set</li>
      </ul>
    </div>
  </div>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML, features=FEATURES, pred=None, prob=None, prob_percent=None, best_k=None)


@app.route("/train", methods=["GET"])
def train_page():
    ensure_trained()
    return render_template_string(
        HTML,
        features=FEATURES,
        pred=None,
        prob=None,
        prob_percent=None,
        best_k=BEST_K,
        cm=CM,
        report=METRICS_TEXT,
        plot_b64=PLOT_B64,
    )


@app.route("/predict", methods=["POST"])
def predict():
    ensure_trained()

    values = []
    for f in FEATURES:
        raw = request.form.get(f, "").strip()
        if raw == "":
            return "Ada input kosong. Balik dan isi semuanya.", 400
        values.append(float(raw))

    X_new = np.array(values).reshape(1, -1)
    X_new_scaled = SCALER.transform(X_new)

    pred = int(MODEL.predict(X_new_scaled)[0])

    # KNN classifier punya predict_proba kalau weights='uniform' (default) dan task klasifikasi
    prob = None
    prob_percent = None
    if hasattr(MODEL, "predict_proba"):
        p = MODEL.predict_proba(X_new_scaled)[0]
        # p[1] = peluang Outcome=1
        prob = float(p[1])
        prob_percent = round(prob * 100, 1)

    return render_template_string(
        HTML,
        features=FEATURES,
        pred=pred,
        prob=round(prob, 4) if prob is not None else "N/A",
        prob_percent=prob_percent,
        best_k=BEST_K,
        cm=CM,
        report=METRICS_TEXT,
        plot_b64=PLOT_B64,
    )


# Railway akan set PORT. Lokal default 5000
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
