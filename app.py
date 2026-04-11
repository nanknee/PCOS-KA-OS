from flask import Flask, request, jsonify, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

model  = pickle.load(open('pcos_chain_svm.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PCOS Detection — SVM</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Segoe UI', sans-serif;
      background: #0d1117;
      color: #c9d1d9;
      min-height: 100vh;
      padding: 2rem 1rem;
    }

    .container {
      max-width: 820px;
      margin: 0 auto;
    }

    header {
      text-align: center;
      margin-bottom: 2rem;
    }

    header h1 {
      font-size: 1.9rem;
      color: #e6edf3;
      font-weight: 600;
      letter-spacing: -0.5px;
    }

    header p {
      font-size: 0.9rem;
      color: #8b949e;
      margin-top: 0.4rem;
    }

    .card {
      background: #161b27;
      border: 1px solid #30363d;
      border-radius: 12px;
      padding: 1.5rem;
      margin-bottom: 1.5rem;
    }

    .card h2 {
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #8b949e;
      margin-bottom: 1rem;
      padding-bottom: 0.6rem;
      border-bottom: 1px solid #30363d;
    }

    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
    }

    .field {
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
    }

    .field label {
      font-size: 0.82rem;
      color: #8b949e;
    }

    .field input, .field select {
      background: #0d1117;
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 0.55rem 0.75rem;
      color: #e6edf3;
      font-size: 0.9rem;
      transition: border-color 0.2s;
      width: 100%;
    }

    .field input:focus, .field select:focus {
      outline: none;
      border-color: #6C63FF;
    }

    .field .hint {
      font-size: 0.72rem;
      color: #484f58;
    }

    .range-wrap {
      display: flex;
      align-items: center;
      gap: 0.6rem;
    }

    .range-wrap input[type=range] {
      flex: 1;
      accent-color: #6C63FF;
      padding: 0;
      border: none;
      background: none;
    }

    .range-val {
      min-width: 42px;
      text-align: right;
      font-size: 0.9rem;
      color: #6C63FF;
      font-weight: 600;
    }

    .predict-btn {
      width: 100%;
      padding: 0.85rem;
      background: #6C63FF;
      color: white;
      border: none;
      border-radius: 10px;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s, transform 0.1s;
      margin-top: 0.5rem;
    }

    .predict-btn:hover  { background: #7c6af7; }
    .predict-btn:active { transform: scale(0.98); }
    .predict-btn:disabled { background: #30363d; cursor: not-allowed; }

    #results { display: none; }

    .primary-banner {
      border-radius: 10px;
      padding: 1.2rem 1.5rem;
      display: flex;
      align-items: center;
      gap: 1rem;
      margin-bottom: 1rem;
    }

    .primary-banner.positive { background: #2d1b1b; border: 1px solid #f07070; }
    .primary-banner.negative { background: #1b2d1b; border: 1px solid #3ecfa0; }

    .banner-circle {
      width: 52px; height: 52px;
      border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-size: 1.3rem; font-weight: 700;
      flex-shrink: 0;
    }

    .positive .banner-circle { background: #f07070; color: #2d1b1b; }
    .negative .banner-circle { background: #3ecfa0; color: #1b2d1b; }

    .banner-text h3 { font-size: 1.1rem; font-weight: 600; }
    .positive .banner-text h3 { color: #f07070; }
    .negative .banner-text h3 { color: #3ecfa0; }
    .banner-text p  { font-size: 0.82rem; color: #8b949e; margin-top: 0.2rem; }

    .symptoms-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 0.75rem;
    }

    .symptom-item {
      background: #0d1117;
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 0.75rem 1rem;
    }

    .symptom-item .s-label {
      font-size: 0.78rem;
      color: #8b949e;
      margin-bottom: 0.3rem;
    }

    .symptom-item .s-result {
      font-size: 0.95rem;
      font-weight: 600;
      margin-bottom: 0.4rem;
    }

    .likely   { color: #f07070; }
    .unlikely { color: #3ecfa0; }

    .conf-bar {
      height: 5px;
      background: #30363d;
      border-radius: 3px;
      overflow: hidden;
    }

    .conf-fill {
      height: 100%;
      border-radius: 3px;
      transition: width 0.6s ease;
    }

    .conf-pct {
      font-size: 0.7rem;
      color: #484f58;
      margin-top: 0.25rem;
    }

    .disclaimer {
      font-size: 0.75rem;
      color: #484f58;
      text-align: center;
      margin-top: 1rem;
    }

    .spinner {
      display: none;
      width: 18px; height: 18px;
      border: 2px solid rgba(255,255,255,0.3);
      border-top-color: white;
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
      margin: 0 auto;
    }

    @keyframes spin { to { transform: rotate(360deg); } }

    @media (max-width: 560px) {
      .grid { grid-template-columns: 1fr; }
      .symptoms-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
<div class="container">

  <header>
    <h1>PCOS Detection</h1>
    <p>Multi-output SVM classifier — enter clinical values to get predictions</p>
  </header>

  <div class="card">
    <h2>Hormonal Markers</h2>
    <div class="grid">
      <div class="field">
        <label>FSH (mIU/mL)</label>
        <div class="range-wrap">
          <input type="range" id="fsh" min="1" max="20" step="0.1" value="7.2"
                 oninput="document.getElementById('fsh_v').textContent=parseFloat(this.value).toFixed(1)">
          <span class="range-val" id="fsh_v">7.2</span>
        </div>
        <span class="hint">Normal: 3–10 mIU/mL</span>
      </div>
      <div class="field">
        <label>LH (mIU/mL)</label>
        <div class="range-wrap">
          <input type="range" id="lh" min="1" max="40" step="0.1" value="5.1"
                 oninput="document.getElementById('lh_v').textContent=parseFloat(this.value).toFixed(1)">
          <span class="range-val" id="lh_v">5.1</span>
        </div>
        <span class="hint">Normal: 2–15 mIU/mL</span>
      </div>
      <div class="field">
        <label>FSH/LH Ratio</label>
        <div class="range-wrap">
          <input type="range" id="fshlh" min="0.1" max="5" step="0.01" value="1.4"
                 oninput="document.getElementById('fshlh_v').textContent=parseFloat(this.value).toFixed(2)">
          <span class="range-val" id="fshlh_v">1.40</span>
        </div>
        <span class="hint">PCOS risk if LH/FSH &gt; 2</span>
      </div>
      <div class="field">
        <label>AMH (ng/mL)</label>
        <div class="range-wrap">
          <input type="range" id="amh" min="0.1" max="15" step="0.1" value="3.2"
                 oninput="document.getElementById('amh_v').textContent=parseFloat(this.value).toFixed(1)">
          <span class="range-val" id="amh_v">3.2</span>
        </div>
        <span class="hint">Elevated (&gt;4.5) suggests PCOS</span>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>Physical Measurements</h2>
    <div class="grid">
      <div class="field">
        <label>BMI</label>
        <div class="range-wrap">
          <input type="range" id="bmi" min="15" max="45" step="0.1" value="23.5"
                 oninput="document.getElementById('bmi_v').textContent=parseFloat(this.value).toFixed(1)">
          <span class="range-val" id="bmi_v">23.5</span>
        </div>
        <span class="hint">Normal: 18.5–24.9</span>
      </div>
      <div class="field">
        <label>Waist:Hip Ratio</label>
        <div class="range-wrap">
          <input type="range" id="whr" min="0.6" max="1.1" step="0.01" value="0.82"
                 oninput="document.getElementById('whr_v').textContent=parseFloat(this.value).toFixed(2)">
          <span class="range-val" id="whr_v">0.82</span>
        </div>
        <span class="hint">Risk if &gt;0.85</span>
      </div>
      <div class="field">
        <label>Cycle Length (days)</label>
        <div class="range-wrap">
          <input type="range" id="cycle" min="15" max="90" step="1" value="30"
                 oninput="document.getElementById('cycle_v').textContent=this.value">
          <span class="range-val" id="cycle_v">30</span>
        </div>
        <span class="hint">Normal: 21–35 days</span>
      </div>
      <div class="field">
        <label>Cycle Type</label>
        <select id="cycletype">
          <option value="0">Regular</option>
          <option value="1">Irregular</option>
        </select>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>Follicle Data (Ultrasound)</h2>
    <div class="grid">
      <div class="field">
        <label>Follicle No. Left</label>
        <div class="range-wrap">
          <input type="range" id="foll_l" min="0" max="30" step="1" value="4"
                 oninput="document.getElementById('foll_l_v').textContent=this.value">
          <span class="range-val" id="foll_l_v">4</span>
        </div>
        <span class="hint">PCOS: typically &gt;12</span>
      </div>
      <div class="field">
        <label>Follicle No. Right</label>
        <div class="range-wrap">
          <input type="range" id="foll_r" min="0" max="30" step="1" value="5"
                 oninput="document.getElementById('foll_r_v').textContent=this.value">
          <span class="range-val" id="foll_r_v">5</span>
        </div>
        <span class="hint">PCOS: typically &gt;12</span>
      </div>
      <div class="field">
        <label>Avg. Follicle Size L (mm)</label>
        <div class="range-wrap">
          <input type="range" id="fsize_l" min="2" max="25" step="0.5" value="11"
                 oninput="document.getElementById('fsize_l_v').textContent=parseFloat(this.value).toFixed(1)">
          <span class="range-val" id="fsize_l_v">11.0</span>
        </div>
      </div>
      <div class="field">
        <label>Avg. Follicle Size R (mm)</label>
        <div class="range-wrap">
          <input type="range" id="fsize_r" min="2" max="25" step="0.5" value="12"
                 oninput="document.getElementById('fsize_r_v').textContent=parseFloat(this.value).toFixed(1)">
          <span class="range-val" id="fsize_r_v">12.0</span>
        </div>
      </div>
    </div>
  </div>

  <button class="predict-btn" id="btn" onclick="predict()">
    <span id="btn-text">Run Prediction</span>
    <div class="spinner" id="spinner"></div>
  </button>

  <div class="card" id="results">
    <h2>Prediction Results</h2>

    <div id="primary-banner" class="primary-banner">
      <div class="banner-circle" id="banner-circle"></div>
      <div class="banner-text">
        <h3 id="banner-title"></h3>
        <p id="banner-sub"></p>
      </div>
    </div>

    <div style="font-size:0.78rem; color:#8b949e; margin-bottom:0.6rem; font-weight:600; text-transform:uppercase; letter-spacing:0.06em;">
      Associated Symptoms
    </div>
    <div class="symptoms-grid" id="symptoms-grid"></div>

    <p class="disclaimer">For research purposes only. Not a substitute for clinical diagnosis.</p>
  </div>

</div>

<script>
async function predict() {
  const btn     = document.getElementById('btn');
  const btnText = document.getElementById('btn-text');
  const spinner = document.getElementById('spinner');

  btn.disabled    = true;
  btnText.style.display = 'none';
  spinner.style.display = 'block';

  const features = [
    parseFloat(document.getElementById('fsh').value),
    parseFloat(document.getElementById('lh').value),
    parseFloat(document.getElementById('fshlh').value),
    parseFloat(document.getElementById('amh').value),
    parseFloat(document.getElementById('bmi').value),
    parseFloat(document.getElementById('whr').value),
    parseFloat(document.getElementById('cycle').value),
    parseFloat(document.getElementById('cycletype').value),
    parseFloat(document.getElementById('foll_l').value),
    parseFloat(document.getElementById('foll_r').value),
  ];

  try {
    const res  = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features })
    });
    const data = await res.json();
    showResults(data);
  } catch(e) {
    alert('Prediction failed. Make sure the server is running.');
  } finally {
    btn.disabled    = false;
    btnText.style.display = 'block';
    spinner.style.display = 'none';
  }
}

function showResults(data) {
  const preds = data.predictions;
  const probs = data.probabilities;
  const pcos  = preds[0] === 1;

  const banner = document.getElementById('primary-banner');
  const circle = document.getElementById('banner-circle');
  const title  = document.getElementById('banner-title');
  const sub    = document.getElementById('banner-sub');

  banner.className = 'primary-banner ' + (pcos ? 'positive' : 'negative');
  circle.textContent = pcos ? 'Y' : 'N';
  title.textContent  = pcos ? 'PCOS Positive' : 'PCOS Negative';
  sub.textContent    = 'Confidence: ' + (probs[0] * 100).toFixed(1) + '%';

  const symptomLabels = ['Weight Gain', 'Hair Growth', 'Skin Darkening', 'Cycle Irregularity'];
  const grid = document.getElementById('symptoms-grid');
  grid.innerHTML = '';

  for (let i = 1; i < preds.length; i++) {
    const likely = preds[i] === 1;
    const pct    = (probs[i] * 100).toFixed(1);
    const color  = likely ? '#f07070' : '#3ecfa0';
    grid.innerHTML += `
      <div class="symptom-item">
        <div class="s-label">${symptomLabels[i-1]}</div>
        <div class="s-result ${likely ? 'likely' : 'unlikely'}">${likely ? 'Likely' : 'Unlikely'}</div>
        <div class="conf-bar">
          <div class="conf-fill" style="width:${pct}%; background:${color};"></div>
        </div>
        <div class="conf-pct">${pct}% confidence</div>
      </div>`;
  }

  const results = document.getElementById('results');
  results.style.display = 'block';
  results.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    data  = request.json['features']
    X     = scaler.transform([data])
    preds = model.predict(X)[0]
    
    # ClassifierChain predict_proba returns shape (n_samples, n_labels)
    # not a list of arrays like MultiOutputClassifier
    probs_matrix = model.predict_proba(X)[0]  # shape: (n_labels,)
    
    return jsonify({
        'predictions':   [int(p) for p in preds],
        'probabilities': [float(p) for p in probs_matrix]
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)