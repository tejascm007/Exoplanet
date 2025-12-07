# Exoplanet Atmospheric Composition Predictor
## Complete Deep Learning Solution

---

## 📋 Requirements

Create a `requirements.txt` file:

```txt
tensorflow==2.15.0
flask==3.0.0
flask-cors==4.0.0
pandas==2.1.4
numpy==1.24.3
scikit-learn==1.3.2
requests==2.31.0
joblib==1.3.2
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 Step-by-Step Setup

### Step 1: Collect Data

Run the data collector to fetch exoplanet data from NASA:

```bash
python exoplanet_data_collector.py
```

**Output:**
- `exoplanet_dataset.csv` - Dataset with features and synthetic atmospheric labels
- `feature_names.pkl` - List of input features

**Expected time:** 2-3 minutes

---

### Step 2: Train the Model

Train the deep learning model:

```bash
python exoplanet_dl_model.py
```

**Output files:**
- `exoplanet_atmosphere_model.keras` - Trained model
- `scaler.pkl` - Feature scaler
- `feature_names.pkl` - Feature names
- `target_molecules.pkl` - Output molecule names
- `best_model.keras` - Best model checkpoint

**Expected time:** 10-30 minutes (depending on hardware)

**Expected accuracy:** MAE ~2-5% per molecule

---

### Step 3: Start Flask API

```bash
python flask_api_backend.py
```

The API will start at: `http://localhost:5000`

---

## 🔌 API Usage

### Health Check
```bash
curl http://localhost:5000/
```

### Make Prediction

**Request:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pl_name": "Kepler-1239 b",
    "pl_orbper": 5.19104016,
    "pl_rade": 2.9,
    "pl_bmasse": null,
    "pl_orbeccen": null,
    "st_teff": 6564,
    "st_rad": 1.57,
    "st_mass": 1.4,
    "sy_dist": 2140.6,
    "pl_eqt": null
  }'
```

**Response:**
```json
{
  "success": true,
  "planet_name": "Kepler-1239 b",
  "atmospheric_composition": {
    "H2": 72.45,
    "He": 18.32,
    "H2O": 3.21,
    "CO": 0.89,
    "CH4": 2.34,
    "CO2": 1.56,
    "NH3": 0.45,
    "N2": 0.78
  },
  "total_percentage": 100.0,
  "confidence": "medium",
  "missing_features": ["pl_bmasse", "pl_orbeccen", "pl_eqt"],
  "metadata": {
    "model": "Transformer-based Deep Learning",
    "molecules_predicted": 8,
    "input_features": 12
  }
}
```

### Get Model Info
```bash
curl http://localhost:5000/model-info
```

---

## 🔗 n8n Integration

### HTTP Request Node Configuration:

**Method:** `POST`

**URL:** `http://localhost:5000/predict`

**Authentication:** None (local)

**Send Body:** ON

**Body Content Type:** `JSON`

**Body:**
```json
{
  "pl_name": "={{ $json.pl_name }}",
  "pl_orbper": "={{ $json.pl_orbper }}",
  "pl_rade": "={{ $json.pl_rade }}",
  "pl_bmasse": "={{ $json.pl_bmasse }}",
  "pl_orbeccen": "={{ $json.pl_orbeccen }}",
  "st_teff": "={{ $json.st_teff }}",
  "st_rad": "={{ $json.st_rad }}",
  "st_mass": "={{ $json.st_mass }}",
  "sy_dist": "={{ $json.sy_dist }}",
  "pl_eqt": "={{ $json.pl_eqt }}"
}
```

---

## 📊 Model Architecture

### Transformer-Based Neural Network

**Input Layer:**
- 12 features (physical parameters + derived features)

**Architecture:**
- Dense (128 units) + BatchNorm + Dropout
- 2x Transformer Blocks (Multi-head attention)
- Dense (256 units) + BatchNorm + Dropout
- Dense (128 units) + BatchNorm + Dropout
- Dense (64 units) + Dropout
- Output (8 molecules) with Softmax

**Output Layer:**
- 8 atmospheric molecules with percentages (sum = 100%)

**Total Parameters:** ~500K

---

## 🎯 Features Used

### Direct Features:
1. `pl_orbper` - Orbital period (days)
2. `pl_rade` - Planet radius (Earth radii)
3. `pl_bmasse` - Planet mass (Earth masses)
4. `pl_orbeccen` - Orbital eccentricity
5. `st_teff` - Stellar temperature (K)
6. `st_rad` - Stellar radius (Solar radii)
7. `st_mass` - Stellar mass (Solar masses)
8. `sy_dist` - System distance (parsecs)
9. `pl_eqt` - Equilibrium temperature (K)

### Derived Features:
10. `insolation_flux` - Energy received from star
11. `surface_gravity` - Planet surface gravity
12. `stellar_luminosity` - Star brightness

---

## 🧪 Model Performance

**Training Set:** 80% of data  
**Test Set:** 20% of data

**Expected Metrics:**
- MAE: 2-5% per molecule
- MSE: 4-25
- R² Score: 0.85-0.95

**Training Time:**
- CPU: 20-30 minutes
- GPU: 5-10 minutes

---

## 🔧 Troubleshooting

### Issue: "Model file not found"
**Solution:** Run `python exoplanet_dl_model.py` to train the model first

### Issue: "Feature names mismatch"
**Solution:** Ensure all `.pkl` files are from the same training run

### Issue: Port 5000 already in use
**Solution:** Change port in `flask_api_backend.py`:
```python
app.run(host='0.0.0.0', port=5001)
```

### Issue: Low prediction accuracy
**Solution:** 
- Increase training epochs (change `epochs=100` to `epochs=200`)
- Collect more real spectroscopic data if available
- Fine-tune hyperparameters

---

## 🚀 Production Deployment

### Using Gunicorn (recommended):

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 flask_api_backend:app
```

### Using Docker:

Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "flask_api_backend:app"]
```

Build and run:
```bash
docker build -t exoplanet-api .
docker run -p 5000:5000 exoplanet-api
```

---

## 📈 Future Improvements

1. **Use Real Spectroscopic Data:**
   - Integrate actual JWST, HST, or ground-based telescope data
   - Replace synthetic labels with observed spectra

2. **Add More Molecules:**
   - TiO, VO (hot Jupiters)
   - Hazes and clouds
   - Atomic species (Na, K)

3. **Uncertainty Quantification:**
   - Implement Bayesian neural networks
   - Provide confidence intervals

4. **Model Ensemble:**
   - Combine multiple models for better accuracy
   - Use voting or averaging

5. **Real-time Updates:**
   - Auto-fetch new exoplanet discoveries
   - Continuous learning

---

## 📚 References

- NASA Exoplanet Archive: https://exoplanetarchive.ipac.caltech.edu/
- Transformer Architecture: "Attention Is All You Need" (Vaswani et al., 2017)
- Exoplanet Atmospheres: Seager & Deming (2010)

---

## 📝 License

MIT License - Feel free to use and modify

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Better atmospheric models
- Integration with real spectroscopic data
- Advanced neural architectures
- Performance optimization

---

**Created for exoplanet atmospheric composition prediction using deep learning**