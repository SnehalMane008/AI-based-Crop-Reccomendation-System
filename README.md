# KRUSHIMITRA - Smart Crop Recommendation System

**जय जवान जय किसान**  

## Project Overview

KRUSHIMITRA is an AI-powered crop recommendation system designed to help farmers make informed decisions about which crops to plant based on soil composition and environmental factors. The system uses machine learning to analyze inputs such as soil NPK values, pH, temperature, humidity, and rainfall to recommend the most suitable crop for given conditions.

## Features

- **Intelligent Crop Recommendations**: Uses a Random Forest machine learning model to predict suitable crops
- **Support for 22 Crops**: Includes major Indian crops like rice, maize, cotton, various pulses, and fruits
- **Detailed Crop Information**: Provides descriptions and ideal growing conditions for each crop
- **Responsive Design**: Mobile-friendly interface for farmers to access from any device
- **Simple Input Form**: Easy-to-use interface for entering soil and climate parameters
- **API Access**: Programmatic access for integration with other agricultural systems

## Technology Stack

- **Backend**: Python with Flask web framework
- **Machine Learning**: Scikit-learn for model training and predictions
- **Frontend**: HTML, CSS, Bootstrap 5 for responsive design
- **Data Processing**: Pandas and NumPy for data manipulation

## UI Enhancement Details

The UI has been enhanced with an agriculture theme, featuring:

1. **Project Branding**:
   - Changed the name to "KRUSHIMITRA" with the tagline "जय जवान जय किसान"
   - Added information about the student creators: Pranav Ladkat, Snehal Mane, and Pratik Khatke

2. **Visual Improvements**:
   - Agriculture-themed color scheme with earthy greens and browns
   - High-resolution farming backgrounds and imagery
   - Custom fonts for better readability and aesthetics
   - Improved layout and spacing

3. **Content Organization**:
   - Added a project banner showcasing the KRUSHIMITRA brand
   - Restructured the input form for better user experience
   - Enhanced the results page with a more visually appealing presentation
   - Added a team section with student information

4. **Responsive Design**:
   - Ensured all elements are responsive across devices
   - Optimized image sizes and layouts for mobile viewing

## Installation and Setup

1. Clone the repository
2. Install required dependencies:
   ```
   install 3.10.11 python version
   to check version 
   python --version

   python -m pip install -r requirements.txt

   ```
3. Run the application:
   ```
   python app.py
   ```
4. Access the application at `http://localhost:5000`

## Adding Image Assets

Before running the application, you need to add the following high-resolution images to the `static/img` directory:

1. Background patterns and UI elements (see `static/img/README.md` for details)
2. Crop images in the `static/img/crops/` directory
3. Team member photos


## Project Structure

```
crop-recommendation/
├── app.py                # Flask application
├── model.py              # ML model training and prediction
├── static/
│   ├── css/              # Stylesheets
│   ├── js/               # JavaScript files
│   └── img/              # Images and visualizations
├── templates/            # HTML templates
│   ├── base.html         # Base template
│   ├── index.html        # Home page
│   ├── result.html       # Prediction results
│   ├── about.html        # About page
│   ├── data_insights.html # Data visualizations
│   └── error.html        # Error page
└── README.md             # Project documentation
```

## Technologies

- **Backend**: Python, Flask
- **Machine Learning**: scikit-learn, pandas, numpy
- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5
- **Data Visualization**: Matplotlib, Seaborn

## Dataset

The model is trained on a dataset containing 2200 samples across 22 different crop types. Each sample includes:

- Nitrogen (N) content (kg/ha)
- Phosphorus (P) content (kg/ha)
- Potassium (K) content (kg/ha)
- Temperature (°C)
- Relative Humidity (%)
- pH value
- Rainfall (mm)

## Model Performance

- **Accuracy**: 99.55%
- **Cross-Validation Score**: 99.09% (mean of 5-fold CV)
- **Precision (Macro Avg)**: 99.57%
- **Recall (Macro Avg)**: 99.55%
- **F1-Score (Macro Avg)**: 99.55%

## API Usage

You can programmatically access the crop recommendation system through the API:

```python
import requests
import json

url = "http://localhost:5000/api/predict"
data = {
    "nitrogen": 90,
    "phosphorus": 42,
    "potassium": 43,
    "temperature": 20.87,
    "humidity": 82.00,
    "ph": 6.5,
    "rainfall": 202.94
}

response = requests.post(url, json=data)
result = response.json()

print(f"Recommended crop: {result['prediction']['crop']}")
print(f"Confidence: {result['prediction']['confidence']}%")
```

## Future Improvements

- Add more crops to the recommendation system
- Integrate weather API for automatic climate data
- Implement user accounts to save field data
- Develop a mobile application for field use
- Add soil type analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset source: [Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
- Icon assets by [Font Awesome](https://fontawesome.com/)
- Built with [Flask](https://flask.palletsprojects.com/) and [Bootstrap](https://getbootstrap.com/) 
