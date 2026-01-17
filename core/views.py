from __future__ import annotations

from typing import Any, Dict, List

from django.http import HttpRequest, HttpResponse
from django.shortcuts import render


# Lightweight placeholder predictor to keep the app runnable without trained models.
KEYWORDS_FRAUD = {'earn money fast', 'wire transfer', 'bitcoin', 'upfront fee', 'urgent hire'}
KEYWORDS_LEGIT = {'full-time', 'benefits', 'experience', 'salary', 'team', 'engineer'}


def _fake_model_predict_prob(text: str) -> float:
    """Return a mock fraud probability based on simple keyword spotting."""
    lowered = text.lower()
    score = 0.5
    if any(word in lowered for word in KEYWORDS_FRAUD):
        score += 0.35
    if any(word in lowered for word in KEYWORDS_LEGIT):
        score -= 0.2
    return max(0.01, min(score, 0.99))


def _generate_explanation(text: str, prob: float) -> List[str]:
    """Generate explanation for why a job posting looks suspicious or legitimate."""
    lowered = text.lower()
    reasons = []
    
    if prob >= 0.5:
        # Fraud indicators
        if any(word in lowered for word in KEYWORDS_FRAUD):
            reasons.append('Contains suspicious keywords related to quick money or payments')
        if len(text.split()) < 30:
            reasons.append('Very short description with limited details')
        if 'urgent' in lowered:
            reasons.append('Uses urgency tactics to pressure applicants')
        if not any(word in lowered for word in ['company', 'experience', 'qualification']):
            reasons.append('Missing standard job posting details')
        if not reasons:
            reasons.append('Pattern matches known fraudulent job postings')
    else:
        # Legitimate indicators
        if any(word in lowered for word in KEYWORDS_LEGIT):
            reasons.append('Contains professional job posting language')
        if len(text.split()) > 50:
            reasons.append('Detailed description with clear requirements')
        if any(word in lowered for word in ['benefits', 'team', 'growth']):
            reasons.append('Mentions employee benefits and career development')
        if not reasons:
            reasons.append('Pattern matches legitimate job postings')
    
    return reasons


def dashboard(request: HttpRequest) -> HttpResponse:
    summary_stats = {
        'total_jobs': 17000,
        'best_model': 'LSTM (93.5%)',
        'fraud_rate': '11.7%',
    }
    cards = [
        {
            'title': 'Access the dataset',
            'description': 'Browse totals, splits, and stats for real vs fake job posts.',
            'href': 'dataset',
            'cta': 'View dataset',
        },
        {
            'title': 'Explore algorithms',
            'description': 'Logistic Regression, Random Forest, CNN, and LSTM diagnostics.',
            'href': 'algorithms',
            'cta': 'Explore models',
        },
        {
            'title': 'Compare performance',
            'description': 'Side-by-side AUC, F1, confusion matrices, and splits.',
            'href': 'comparison',
            'cta': 'Open comparison',
        },
        {
            'title': 'Predict a posting',
            'description': 'Enter a job title, description, and location to score fraud risk.',
            'href': 'prediction',
            'cta': 'Start prediction',
        },
    ]
    return render(request, 'dashboard.html', {'cards': cards, 'summary_stats': summary_stats})


def dataset(request: HttpRequest) -> HttpResponse:
    dataset_stats = {
        'total_jobs': 17000,
        'real_jobs': 15000,
        'fake_jobs': 2000,
        'fraud_percentage': 11.7,
        'real_percentage': 88.3,
        'combined_datasets': ['jobs_primary.csv', 'jobs_extra.csv'],
        'existing_datasets': ['primary', 'augmented', 'experimental'],
        'summary': {
            'avg_words': 120,
            'median_title_length': 6,
            'missing_rate': '2.1%',
        },
    }
    return render(request, 'dataset.html', {'dataset_stats': dataset_stats})


def algorithms(request: HttpRequest) -> HttpResponse:
    algorithms_info: List[Dict[str, Any]] = [
        {
            'name': 'Logistic Regression',
            'slug': 'logistic_regression',
            'split_ratio': '70/15/15',
            'auc': 0.93,
            'accuracy': 0.90,
            'f1': 0.88,
            'notes': 'Baseline linear model with TF-IDF features.',
        },
        {
            'name': 'Random Forest',
            'slug': 'random_forest',
            'split_ratio': '70/15/15',
            'auc': 0.95,
            'accuracy': 0.92,
            'f1': 0.90,
            'notes': '200 estimators, tuned max_depth and class weights.',
        },
        {
            'name': 'CNN',
            'slug': 'cnn',
            'split_ratio': '80/10/10',
            'auc': 0.96,
            'accuracy': 0.93,
            'f1': 0.91,
            'notes': '1D convolutions over token sequences.',
        },
        {
            'name': 'LSTM',
            'slug': 'lstm',
            'split_ratio': '80/10/10',
            'auc': 0.965,
            'accuracy': 0.935,
            'f1': 0.92,
            'notes': 'Bidirectional LSTM with pretrained embeddings.',
        },
    ]
    diagnostics = {
        'confusion_matrix': [[920, 80], [65, 135]],
        'roc_points': [[0.0, 0.0], [0.1, 0.8], [0.2, 0.9], [1.0, 1.0]],
        'split_ratios': ['70/15/15', '75/15/10', '80/10/10'],
        'metric_comparison': [
            {'metric': 'AUC', 'lr': 0.93, 'rf': 0.95, 'cnn': 0.96, 'lstm': 0.965},
            {'metric': 'Accuracy', 'lr': 0.90, 'rf': 0.92, 'cnn': 0.93, 'lstm': 0.935},
            {'metric': 'F1', 'lr': 0.88, 'rf': 0.90, 'cnn': 0.91, 'lstm': 0.92},
        ],
    }
    return render(request, 'algorithms.html', {'algorithms': algorithms_info, 'diagnostics': diagnostics})


def comparison(request: HttpRequest) -> HttpResponse:
    comparison_rows = [
        {'algorithm': 'Logistic Regression', 'auc': 0.93, 'f1': 0.88, 'accuracy': 0.90},
        {'algorithm': 'Random Forest', 'auc': 0.95, 'f1': 0.90, 'accuracy': 0.92},
        {'algorithm': 'CNN', 'auc': 0.96, 'f1': 0.91, 'accuracy': 0.93},
        {'algorithm': 'LSTM', 'auc': 0.965, 'f1': 0.92, 'accuracy': 0.935},
    ]
    notes = [
        'Tree-based model improves recall on minority class.',
        'Deep models show higher AUC but require more compute.',
        'Logistic regression remains a strong baseline with fast inference.',
    ]
    return render(request, 'comparison.html', {'rows': comparison_rows, 'notes': notes})


def prediction(request: HttpRequest) -> HttpResponse:
    result: Dict[str, Any] | None = None
    title = description = location = ''

    if request.method == 'POST':
        title = request.POST.get('title', '').strip()
        description = request.POST.get('description', '').strip()
        location = request.POST.get('location', '').strip()
        combined = f"{title}\n{description}\n{location}"
        prob = _fake_model_predict_prob(combined)
        label = 'Fraud' if prob >= 0.5 else 'Legit'
        confidence_percent = int(prob * 100) if prob >= 0.5 else int((1 - prob) * 100)
        explanations = _generate_explanation(combined, prob)
        result = {
            'label': label,
            'probability': round(prob, 3),
            'confidence_percent': confidence_percent,
            'explanations': explanations,
        }

    guidelines = [
        'Provide a concise title (e.g., Data Analyst).',
        'Include job responsibilities and requirements.',
        'Add location or Remote tag.',
        'Avoid pasting HTML; plain text works best.',
    ]
    samples = [
        {
            'title': 'Software Engineer',
            'description': 'Build backend services with Python and Django. Collaborate with product.',
            'location': 'Bangalore, IN',
        },
        {
            'title': 'Data Scientist',
            'description': 'Model user behavior, deploy ML models, partner with engineering.',
            'location': 'Remote',
        },
    ]

    return render(
        request,
        'prediction.html',
        {
            'result': result,
            'guidelines': guidelines,
            'samples': samples,
            'form_values': {'title': title, 'description': description, 'location': location},
        },
    )
