
# Sentiment Analysis Project

## Description
This project uses a pre-trained BERT model to classify text into three sentiment categories: Positive, Neutral, and Negative.

## Requirements
- Python 3.7+
- PyTorch
- Transformers library

## Installation
1. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure the pre-trained model is placed in the `./model` directory.

## Usage
Run the script to analyze sentiment of sample comments:
```bash
python sentiment_analysis.py
```

You can modify the `test_comments` list in the script to test with your own data.

## Files
- `sentiment_analysis.py`: The main script for sentiment analysis.
- `requirements.txt`: Python dependencies for the project.
- `README.md`: This file with project documentation.
