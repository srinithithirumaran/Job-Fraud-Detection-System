# Job Fraud Detection Web Application

A Django-based web application for job posting fraud detection using machine learning. Analyze datasets, explore four ML algorithms, compare performance, and predict fraud risk for individual postings.

## Features

- **Dashboard**: Quick access to all major sections with intuitive cards
- **Dataset Page**: View dataset statistics (total, real, fake jobs) and data splits
- **Algorithms Page**: Explore four ML models
  - Logistic Regression (TF-IDF baseline)
  - Random Forest (ensemble method)
  - CNN (1D convolutional neural network)
  - LSTM (recurrent neural network)
  - Displays: confusion matrix, ROC curves, AUC, F1 scores, data distributions
- **Comparison Page**: Side-by-side performance analysis of all models
- **Prediction Page**: Input a job posting to predict fraud probability
  - Fields: title, description, location
  - Returns: fraud/legit label and confidence score
  - Includes guidelines and sample postings
 
## 📸 Screenshots

![Dashboard](Screenshots/Dashboard.jpeg)
![Dataset](Screenshots/Dataset.jpeg)
![Algorithms](Screenshots/Algorithms.jpeg)
![Comparison](Screenshots/Comparison.jpeg)
![Prediction](Screenshots/Prediction.jpeg)


## Project Structure

```
jobfraud/
├── manage.py                     # Django management script
├── jobfraud/                     # Project configuration
│   ├── settings.py              # Django settings (INSTALLED_APPS, templates, static paths)
│   ├── urls.py                  # Root URL routing
│   ├── wsgi.py                  # WSGI app
│   └── asgi.py
├── core/                         # Main app
│   ├── views.py                 # View functions (dashboard, dataset, algorithms, etc.)
│   ├── urls.py                  # App URL patterns
│   ├── models.py                # Django models (optional)
│   ├── admin.py                 # Django admin registration
│   └── apps.py
├── templates/                    # HTML templates
│   ├── base.html                # Base layout with sidebar navigation
│   ├── dashboard.html           # Dashboard with cards
│   ├── dataset.html             # Dataset stats
│   ├── algorithms.html          # Algorithm details and metrics
│   ├── comparison.html          # Model comparison
│   └── prediction.html          # Fraud prediction form
├── static/                       # Static files
│   └── css/
│       └── styles.css           # Custom CSS styling
├── ml/                          # Machine learning
│   ├── train.py                 # Training script (Logistic Regression, Random Forest, CNN, LSTM)
│   ├── data/                    # Dataset directory (place jobs_dataset.csv here)
│   ├── models/                  # Trained model artifacts (joblib, .h5)
│   ├── metrics/                 # Model metrics JSON files
│   └── plots/                   # Generated plots (confusion matrix, ROC, etc.)
└── .github/                      # GitHub-specific
    └── copilot-instructions.md  # Copilot setup instructions
```

## Installation & Setup

### 1. Clone or navigate to the project

```bash
cd "C:\Users\kanmani dhaya\New folder"
```

### 2. Activate virtual environment

The project uses a Python virtual environment. Activate it:

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
pip install django djangorestframework scikit-learn pandas numpy matplotlib seaborn plotly joblib tensorflow-cpu
```

### 4. Run migrations

```bash
python manage.py migrate
```

### 5. Start the development server

```bash
python manage.py runserver
```

The app will be available at `http://localhost:8000`

## Pages Overview

### Dashboard (`/`)
- Four cards linking to main sections
- Quick navigation to dataset, algorithms, comparison, and prediction

### Dataset (`/dataset/`)
- Total jobs: **17,000** (placeholder—replace with actual count)
- Real jobs vs fake jobs breakdown
- Dataset statistics (avg words, median title length, etc.)
- List of combined datasets and dataset variants

### Algorithms (`/algorithms/`)
- **Logistic Regression**: Fast, interpretable baseline
- **Random Forest**: Ensemble, handles non-linear patterns
- **CNN**: Convolutional layers for text, feature extraction
- **LSTM**: Recurrent network with bidirectional processing

For each model, displays:
- Split ratio (e.g., 70/15/15 for train/val/test)
- AUC, Accuracy, F1 score
- Confusion matrix
- ROC curve (placeholder)

### Comparison (`/comparison/`)
- Tabular comparison of AUC, F1, Accuracy across all four models
- Key insights (e.g., "tree-based models improve recall on minority class")
- Placeholder for Plotly/Chart.js comparison chart

### Prediction (`/prediction/`)
- Form with three inputs: job title, description, location
- Returns fraud probability and risk label (Fraud or Legit)
- Right sidebar with:
  - How-to guidelines
  - Sample job postings (Data Scientist, Software Engineer)

## Training & ML Pipeline

### Load Your Dataset

1. Place your CSV file at `ml/data/jobs_dataset.csv`
2. Expected columns:
   - `title`: Job title string
   - `description`: Job description text
   - `location`: Location string
   - `label`: 0 (legitimate) or 1 (fraudulent)

### Run Training Script

```bash
python ml/train.py
```

This script will:
1. Load and split the dataset (70/15/15 by default)
2. Preprocess text (TF-IDF vectorization)
3. Train Logistic Regression and Random Forest
4. Export models and metrics to `ml/models/` and `ml/metrics/`
5. Print evaluation metrics

### Integrate Trained Models into Views

After training, update [core/views.py](core/views.py) to load saved models:

```python
import joblib

lr_model = joblib.load('ml/models/logistic_regression.pkl')
vectorizer = joblib.load('ml/models/vectorizer.pkl')

def predict(title, description, location):
    combined = f"{title} {description} {location}"
    X = vectorizer.transform([combined])
    prob = lr_model.predict_proba(X)[0, 1]
    return "Fraud" if prob > 0.5 else "Legit", prob
```

## Customization

### Add CNN & LSTM Models

The `ml/train.py` contains placeholder functions. To implement:

1. Use TensorFlow/Keras for neural networks
2. Define architecture (embeddings → Conv/LSTM → Dense)
3. Train on GPU if available (TensorFlow handles this)
4. Save as `.h5` files

Example outline:
```python
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_len),
    keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu'),
    keras.layers.GlobalMaxPooling1D(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)
model.save('ml/models/cnn_model.h5')
```

### Update Metrics in Views

After training, replace placeholder values in `algorithms` and `comparison` views:

```python
# Load from JSON files saved during training
import json
with open('ml/metrics/logistic_regression.json') as f:
    lr_metrics = json.load(f)
```

### Styling

Modify [static/css/styles.css](static/css/styles.css) to customize colors, fonts, and layout.

## Development Notes

- **Framework**: Django 5.0 with Bootstrap 5.3.3
- **Database**: SQLite (default)
- **ML Libraries**: scikit-learn, pandas, numpy, TensorFlow/Keras (optional for CNN/LSTM)
- **Visualization**: Plotly, Matplotlib, Seaborn (plots saved as images/JSON)
- **Text Processing**: TF-IDF (sklearn), Tokenizer (Keras)

## Troubleshooting

### Port 8000 already in use
```bash
python manage.py runserver 8001
```

### Import errors for ML packages
```bash
pip install --upgrade scikit-learn pandas numpy tensorflow-cpu
```

### Dataset not found
Place your CSV at `ml/data/jobs_dataset.csv` and re-run `ml/train.py`.

### Migrations errors
```bash
python manage.py makemigrations
python manage.py migrate
```

## Future Enhancements

1. **Database Models**: Store predictions and user history
2. **API Endpoints**: REST API for programmatic access
3. **Real-time Metrics**: Dashboard with live model performance tracking
4. **Explainability**: SHAP/LIME to explain individual predictions
5. **Deployment**: Docker, AWS/GCP cloud deployment
6. **Authentication**: User login and role-based access

## License

This project is for educational purposes. Adjust licensing as needed.

## Support

For issues or questions, refer to [Django documentation](https://docs.djangoproject.com/) and [scikit-learn guides](https://scikit-learn.org/stable/user_guide.html).





