# app.py
from flask import Flask, render_template, request, jsonify, session
from flask_babel import Babel, gettext as _, lazy_gettext as _l,get_locale as babel_get_locale  # added gettext as _
import pandas as pd
import numpy as np
import joblib
import os
from model import predict_crop, train_crop_recommendation_model

app = Flask(__name__)

# === Flask-Babel basic config ===
app.config['SECRET_KEY'] = 'change-me'  # required for storing chosen language in session
app.config['BABEL_DEFAULT_LOCALE'] = 'en'
app.config['BABEL_TRANSLATION_DIRECTORIES'] = 'translations'

# Language names are potentially shown in UI; mark with lazy_gettext at module scope
LANGUAGES = {'en': _l('English'), 'hi': _l('Hindi'), 'mr': _l('Marathi')}

babel = Babel()

def select_locale():  # renamed to avoid name clash
    lang = request.args.get('lang')
    if lang in LANGUAGES:
        session['lang'] = lang
        return lang
    return session.get('lang') or request.accept_languages.best_match(LANGUAGES.keys()) or 'en'

# register selector in FB 4.x
babel.init_app(app, locale_selector=select_locale)

@app.context_processor
def inject_get_locale():
    return {'get_locale': lambda: str(babel_get_locale())}

# Crop descriptions for the UI (wrapped with lazy _l so translation happens per request)
CROP_INFO = {
    'rice': {
        'description': _l('''Best time: Kharif (monsoon); needs warm weather and good water.
                            Soil and water: Grows well in clay/loam soil; keep field level; avoid very deep water.
                            Nutrients: Use balanced NPK; do not overuse urea.
                            Major diseases: Blast, bacterial leaf blight, sheath blight.
                            Simple solutions: Use resistant seed, treat seed, keep proper spacing, drain excess water; spray as per local agri advice if disease crosses threshold.
                            Profit idea: Gross = yield per acre × price (MSP/local mandi). Control disease early to protect yield and profit.'''),
        'ideal_conditions': _l('N: 80-100, P: 40-60, K: 40-45, Temperature: 20-25°C, Humidity: 80-85%, pH: 6.0-7.0, Rainfall: 200-250mm')
    },
    'maize': {
        'description': _l('''Best time: Kharif and also Rabi in many areas.
                            Soil and water: Needs well-drained soil; avoid waterlogging.
                            Nutrients: Starter fertilizer at sowing; top-dress nitrogen at knee-height and tasseling.
                            Major pest/disease: Fall armyworm, leaf blights.
                            Simple solutions: Pheromone traps, destroy egg masses, neem-based sprays early; choose tolerant hybrids; rotate crops.
                            Profit idea: Keep fields weed-free early; timely protection against fall armyworm saves cobs and boosts income.'''),
        'ideal_conditions': _l('N: 80-100, P: 50-80, K: 40-60, Temperature: 22-30°C, Humidity: 50-80%, pH: 5.5-7.0, Rainfall: 80-100mm')
    },
    'jute': {
        'description': _l('''Best time: Pre-monsoon to early monsoon.
                            Soil and water: Moist, well-drained soil; avoid standing water.
                            Nutrients: Basal NPK; keep weeds low.
                            Major diseases: Stem/soft rot in waterlogged fields.
                            Simple solutions: Good drainage, wider spacing where needed, seed treatment, field sanitation.
                            Profit idea: Proper retting in clean water gives better fibre grade and price.'''),
        'ideal_conditions': _l('N: 50-80, P: 40-60, K: 40-50, Temperature: 25-35°C, Humidity: 70-90%, pH: 6.0-7.5, Rainfall: 150-200mm')
    },
    'cotton': {
        'description': _l('''Best time: Kharif (rainfed/irrigated depending on region).
                            Soil and water: Deep, well-drained soil; avoid excess nitrogen.
                            Nutrients: Balanced NPK with micronutrients; split nitrogen.
                            Major pests/diseases: Pink bollworm, whitefly, boll rot.
                            Simple solutions: Pheromone traps, timely sowing, remove “rosette” flowers, avoid early pyrethroids, use need-based sprays; keep field clean.
                            Profit idea: Protect bolls during peak fruiting; better lint and more pickings improve returns.'''),
        'ideal_conditions': _l('N: 80-120, P: 40-60, K: 40-60, Temperature: 20-30°C, Humidity: 50-80%, pH: 5.5-8.0, Rainfall: 80-120mm')
    },
    'orange': {
        'description': _l('''Best time: Orchard crop; plan planting in mild season.
                            Soil and water: Well-drained soil; do not let water stand near trunk.
                            Nutrients: Regular manure + NPK + micronutrients; maintain pH around slightly acidic to neutral.
                            Major diseases: Citrus canker, greening (HLB), gummosis.
                            Simple solutions: Use clean planting material, prune for airflow, copper sprays for canker, manage psyllid (HLB vector), protect trunk from injury and rot.
                            Profit idea: Good canopy and timely sprays increase healthy fruits and market price.'''),
        'ideal_conditions': _l('N: 30-60, P: 30-50, K: 30-50, Temperature: 15-30°C, Humidity: 40-70%, pH: 5.5-7.0, Rainfall: 80-150mm')
    },
    'apple': {
        'description': _l('''Best time: Temperate zones only.
                            Soil and water: Well-drained loam; avoid waterlogging.
                            Nutrients: Balanced NPK; prune for light and air.
                            Major diseases: Scab, powdery mildew, fire blight.
                            Simple solutions: Orchard sanitation, pruning, protectant/systemic fungicides at key stages as per advisory.
                            Profit idea: Higher “A-grade” fruit gives better price; disease control at right time improves packout.'''),
        'ideal_conditions': _l('N: 20-40, P: 20-40, K: 30-50, Temperature: 15-25°C, Humidity: 40-70%, pH: 5.5-6.5, Rainfall: 80-120mm')
    },
    'watermelon': {
        'description': _l('''Best time: Warm season; avoid heavy rains during fruiting.
                            Soil and water: Sandy loam; raised beds and mulch help; drip irrigation is best.
                            Nutrients: Balanced NPK with extra potassium for sweetness.
                            Major diseases: Anthracnose, downy mildew, Fusarium wilt.
                            Simple solutions: Rotate (do not plant after cucurbits), resistant varieties, field sanitation, protectant/systemic fungicides when needed.
                            Profit idea: Time harvest for peak demand; clean, sweet fruits fetch better price.'''),
        'ideal_conditions': _l('N: 20-40, P: 30-50, K: 30-50, Temperature: 22-32°C, Humidity: 50-80%, pH: 6.0-7.0, Rainfall: 40-70mm')
    },
    'grapes': {
        'description': _l('''Best time: Perennial; pruning and training are key.
                            Soil and water: Deep, well-drained loam; avoid high salts; maintain pH near neutral.
                            Nutrients: Balanced feeding with micronutrients; regulate canopy.
                            Major diseases: Powdery mildew, downy mildew, anthracnose, bunch rot.
                            Simple solutions: Keep canopy open, remove infected parts, use sulphur/copper and systemic sprays as per schedule; maintain bunch hygiene.
                            Profit idea: Better brix, berry size, and appearance give higher rates; disease-free bunches raise packout and profit.'''),
        'ideal_conditions': _l('N: 20-40, P: 20-40, K: 30-50, Temperature: 15-25°C, Humidity: 60-70%, pH: 6.0-7.0, Rainfall: 60-100mm')
    },
    'mango': {
        'description': _l('''Best time: Tropical/sub-tropical; plant in mild weather.
                            Soil and water: Deep, well-drained soil; avoid waterlogging around trunk.
                            Nutrients: Manure + NPK; zinc/boron if deficient; prune after harvest.
                            Major diseases: Anthracnose, powdery mildew, malformation.
                            Simple solutions: Prune for airflow, orchard sanitation, copper or mancozeb/carbendazim sprays at flowering/fruit set, sulphur for mildew; hot water dip post-harvest if possible.
                            Profit idea: Clean panicles and healthy fruits reduce post-harvest losses and raise income.'''),
        'ideal_conditions': _l('N: 20-40, P: 20-40, K: 30-50, Temperature: 24-30°C, Humidity: 50-80%, pH: 5.5-7.5, Rainfall: 70-150mm')
    },
    'banana': {
        'description': _l('''Best time: Year-round in many areas with irrigation.
                            Soil and water: Fertile, well-drained soil; steady moisture is important.
                            Nutrients: High feeder; follow fertigation schedule; add organic matter.
                            Major diseases/pests: Fusarium wilt, Sigatoka leaf spots, bunchy top virus.
                            Simple solutions: Use disease-free tissue culture plants, sanitize tools, good drainage, scheduled leaf-spot sprays, remove infected mats.
                            Profit idea: Uniform bunch size and clean peels get better rate; careful harvest and packing improve returns.'''),
        'ideal_conditions': _l('N: 80-120, P: 20-40, K: 50-70, Temperature: 20-30°C, Humidity: 70-90%, pH: 5.5-7.0, Rainfall: 120-180mm')
    },
    'chickpea': {
        'description': _l('''Best time: Rabi (post-monsoon).
                            Soil and water: Grows well in light to medium soils; avoid waterlogging.
                            Nutrients: Starter fertilizer and sulphur help; inoculate seed with Rhizobium where needed.
                            Major diseases: Fusarium wilt, Ascochyta blight.
                            Simple solutions: Seed treatment, resistant varieties, crop rotation, timely fungicide if disease appears.
                            Profit idea: MSP helps as price floor; timely sowing and wilt management protect yield and profit.'''),
        'ideal_conditions': _l('N: 20-40, P: 40-60, K: 20-40, Temperature: 15-30°C, Humidity: 40-70%, pH: 6.0-8.0, Rainfall: 60-100mm')
    },
    'coffee': {
        'description': _l('''Best time: Grown in hilly, shaded, humid areas.
                            Soil and water: Well-drained, rich soil; regulate shade.
                            Nutrients: Regular manure + NPK; prune for light.
                            Major diseases/pests: Leaf rust, berry diseases, coffee berry borer.
                            Simple solutions: Sanitation, pruning, copper/systemic sprays at the right times, trap/monitor borers.
                            Profit idea: Good picking and clean processing increase grade and price.'''),
        'ideal_conditions': _l('N: 80-120, P: 20-40, K: 20-40, Temperature: 15-28°C, Humidity: 50-70%, pH: 6.0-7.0, Rainfall: 150-200mm')
    }
}

@app.route('/')
def home():
    # Get language from query parameter or session
    lang = request.args.get('lang')
    if lang and lang in LANGUAGES:
        session['lang'] = lang
    # Page content is translated in templates
    return render_template('index_blink.html')
from monitoring.monitor import PredictionLogger

prediction_logger = PredictionLogger()

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # 1) Extract values from form
            N = float(request.form['nitrogen'])
            P = float(request.form['phosphorus'])
            K = float(request.form['potassium'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])

            # 2) Make prediction
            crop_name, confidence = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
            predicted_crop = crop_name.lower()

            prediction_logger.log(
                inputs={"N": N, "P": P, "K": K, "temperature": temperature,
                
                "humidity": humidity, "ph": ph, "rainfall": rainfall},
                prediction=crop_name,
                confidence=confidence
            )
            # 3) Get crop information with translated fallbacks (use gettext inside handlers)
            crop_description = CROP_INFO.get(predicted_crop, {}).get('description', _('No description available.'))
            ideal_conditions = CROP_INFO.get(predicted_crop, {}).get('ideal_conditions', _('Information not available.'))

            from utils.roi import compute_roi, is_msp_crop

            # 4) Optional ROI inputs
            price_per_quintal = request.form.get("price_per_quintal", type=float)
            price_per_kg = request.form.get("price_per_kg", type=float)
            yield_q_per_acre = request.form.get("yield_q_per_acre", type=float)
            yield_kg_per_acre = request.form.get("yield_kg_per_acre", type=float)
            total_cost_per_acre = request.form.get("total_cost_per_acre", type=float) or 0.0

            roi = compute_roi(
                predicted_crop,
                price_per_quintal=price_per_quintal,
                price_per_kg=price_per_kg,
                yield_q_per_acre=yield_q_per_acre,
                yield_kg_per_acre=yield_kg_per_acre,
                total_cost_per_acre=total_cost_per_acre
            )

            # 5) Prepare response payload for template
            result = {
                'crop': crop_name.title(),
                'confidence': round(confidence, 2),
                'roi': roi,  # Pass the ROI data to the template
                'description': crop_description,
                'ideal_conditions': ideal_conditions,
                'image': f"{predicted_crop}.jpeg"  # <-- change .jpg to .jpeg
            }

# ...rest of your code...

            return render_template(
                'result_blink.html',
                result=result,
                input_data={
                    'nitrogen': N,
                    'phosphorus': P,
                    'potassium': K,
                    'temperature': temperature,
                    'humidity': humidity,
                    'ph': ph,
                    'rainfall': rainfall
                },
                roi=roi,
                is_msp=is_msp_crop(predicted_crop)
            )

        except Exception as e:
            # Use a translated generic error label; actual exception is shown in template
            return render_template('error.html', error=str(e))

    # Fallback GET
    return render_template('index.html')

@app.route('/about')
def about():
    lang = request.args.get('lang')
    if lang and lang in LANGUAGES:
        session['lang'] = lang
    return render_template('about_blink.html')

@app.route('/insights')
def data_insights():
    lang = request.args.get('lang')
    if lang and lang in LANGUAGES:
        session['lang'] = lang
    return render_template('data_insights_blink.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for ML predictions (can be triggered by voice or form)"""
    try:
        data = request.get_json()
        
        # Extract parameters
        N = float(data.get('nitrogen', 0))
        P = float(data.get('phosphorus', 0))
        K = float(data.get('potassium', 0))
        temperature = float(data.get('temperature', 0))
        humidity = float(data.get('humidity', 0))
        ph = float(data.get('ph', 0))
        rainfall = float(data.get('rainfall', 0))
        
        # Get prediction from ML model
        predicted_crop, confidence = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
        
        # Get crop information
        crop_description = CROP_INFO.get(predicted_crop, {}).get('description', _('No description available'))
        ideal_conditions = CROP_INFO.get(predicted_crop, {}).get('ideal_conditions', _('No conditions available'))
        
        return jsonify({
            'success': True,
            'crop': predicted_crop,
            'confidence': round(confidence, 2),
            'description': str(crop_description),
            'ideal_conditions': str(ideal_conditions),
            'input_data': {
                'nitrogen': N,
                'phosphorus': P,
                'potassium': K,
                'temperature': temperature,
                'humidity': humidity,
                'ph': ph,
                'rainfall': rainfall
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/recommend')
def recommend():
    lang = request.args.get('lang')
    if lang and lang in LANGUAGES:
        session['lang'] = lang
    return render_template('recommend_blink.html')

if __name__ == '__main__':
    # Check if model exists, if not train it
    if not os.path.exists('crop_recommendation_model.pkl'):
        print("Model not found. Training new model...")
        train_crop_recommendation_model()
    
    app.run(debug=True, port=5000)
