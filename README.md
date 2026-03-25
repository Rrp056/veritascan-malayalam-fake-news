# VeriTaScan - Malayalam Fake News Detector

A machine learning-powered web application for detecting fake news in Malayalam language using TF-IDF vectorization and Multinomial Naive Bayes classification.

## Features

- **Single Text Detection**: Analyze individual Malayalam news articles
- **Batch Processing**: Upload CSV files for bulk classification
- **Model Performance Metrics**: View precision, recall, and F1-scores
- **F-Score Dashboard**: Dedicated visualization of F-scores with interactive F-beta calculator
- **Dataset Explorer**: Browse and filter the training dataset

## Technology Stack

- **Frontend**: Streamlit
- **ML Framework**: scikit-learn
- **Data Processing**: pandas, numpy
- **Language**: Python 3.9+

## Model Details

- **Algorithm**: Multinomial Naive Bayes
- **Features**: TF-IDF with character n-grams (2-4 grams)
- **Training Data**: Malayalam news articles (true/fake labeled)
- **Performance**: ~XX% accuracy on test set

## Local Development

1. Clone/download the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```
4. Open `http://localhost:8501`

## Deployment

This app is deployed on Streamlit Cloud. The live version is available at:
[https://veritascan-malayalam-fake-news.streamlit.app/](https://veritascan-malayalam-fake-news.streamlit.app/)

## Files

- `app.py` - Main Streamlit application
- `ma_true.csv` - True news dataset
- `ma_fake.csv` - Fake news dataset
- `requirements.txt` - Python dependencies
- `packages.txt` - Streamlit Cloud dependencies

## Usage

1. **Detect Tab**: Paste Malayalam news text and get instant classification
2. **Batch Tab**: Upload CSV with 'text' column for bulk analysis
3. **Explore Tab**: View model metrics and explore the dataset

## License

MIT License