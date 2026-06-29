import os
import pickle
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the trained model safely
with open('crop_recommendation_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# Fully loaded UI Template with original styling and automatic scroll fixed for Render deployment
html_code = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🌾 Crop Advisory Agent</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <style>
        body { background-color: #f4f7f6; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .card { border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
        .btn-success { background-color: #2ec4b6; border: none; }
        .btn-success:hover { background-color: #0cb0a1; }
        .result-box { display: none; border-left: 5px solid #2ec4b6; background: #e6f9f7; }
        .white-space-pre { white-space: pre-line; }
    </style>
</head>
<body>

<div class="container py-5">
    <div class="row justify-content-center">
        <div class="col-md-8">
            <div class="card p-4 mb-4">
                <h2 class="text-center text-success mb-4">🌱 Crop Recommendation Agent</h2>
                <p class="text-muted text-center">Enter soil nutrients and climate conditions to get the best crop prediction.</p>
                <hr>
                
                <form id="predictionForm" class="row g-3">
                    <div class="col-md-4">
                        <label class="form-label">Nitrogen (N)</label>
                        <input type="number" class="form-control" name="N" value="90" required>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label">Phosphorus (P)</label>
                        <input type="number" class="form-control" name="P" value="42" required>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label">Potassium (K)</label>
                        <input type="number" class="form-control" name="K" value="43" required>
                    </div>
                    <div class="col-md-6">
                        <label class="form-label">Temperature (°C)</label>
                        <input type="number" step="any" class="form-control" name="temp" value="21" required>
                    </div>
                    <div class="col-md-6">
                        <label class="form-label">Humidity (%)</label>
                        <input type="number" step="any" class="form-control" name="humidity" value="82" required>
                    </div>
                    <div class="col-md-6">
                        <label class="form-label">Soil pH</label>
                        <input type="number" step="any" class="form-control" name="ph" value="6.5" required>
                    </div>
                    <div class="col-md-6">
                        <label class="form-label">Rainfall (mm)</label>
                        <input type="number" step="any" class="form-control" name="rainfall" value="200" required>
                    </div>
                    
                    <div class="col-12 text-center mt-4">
                        <button type="submit" id="submitBtn" class="btn btn-success btn-lg px-5">Predict Best Crop 🚀</button>
                    </div>
                </form>
            </div>

            <div id="resultCard" class="card p-4 result-box">
                <h3 class="text-dark">🌾 Recommendation: <span id="cropResult" class="text-success fw-bold"></span></h3>
                <div id="planResult" class="text-muted mt-2 white-space-pre"></div>
            </div>

        </div>
    </div>
</div>

<script>
document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    const btn = document.getElementById('submitBtn');
    btn.innerText = "Processing... ⏳";
    btn.disabled = true;

    const formData = new FormData(this);
    const data = Object.fromEntries(formData);
    
    try {
        // Appends predict to relative path seamlessly for Render multi-environment hosting
        const response = await fetch('https://crop-recommendation-agent.onrender.com/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        document.getElementById('cropResult').innerText = result.crop;
        document.getElementById('planResult').innerHTML = result.plan;
        document.getElementById('resultCard').style.display = 'block';
        document.getElementById('resultCard').scrollIntoView({ behavior: 'smooth' });
    } catch (error) {
        alert("⚠️ Deployment Error: Communication with the server failed.");
    } finally {
        btn.innerText = "Predict Best Crop 🚀";
        btn.disabled = false;
    }
});
</script>

</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(html_code)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([[
        float(data['N']), float(data['P']), float(data['K']),
        float(data['temp']), float(data['humidity']),
        float(data['ph']), float(data['rainfall'])
    ]])
    
    crop = rf_model.predict(features)[0].upper()
    
    plan = f"""
    <strong>[CULTIVATION PLAN FOR {crop}]</strong><br>
    • Optimal Soil Maintenance: Keep pH levels stable around {float(data['ph']):.1f}.<br>
    • Water Resource Allocation: Based on rainfall ({data['rainfall']}mm), setup precise irrigation.<br>
    • Fertilizer Strategy: Adjust fields based on inputs N:{data['N']}, P:{data['P']}, K:{data['K']}.
    """
    return jsonify({'crop': crop, 'plan': plan})

if __name__ == '__main__':
    # Dynamic port configuration mandatory for Render platform stability
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
