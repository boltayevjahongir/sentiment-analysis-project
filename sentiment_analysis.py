
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the trained tokenizer and model
tokenizer = BertTokenizer.from_pretrained('./model')
model = BertForSequenceClassification.from_pretrained('./model')

# Sentiment analysis function
def analyze_sentiment(text):
    """Analyze the sentiment of a given text."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    sentiment = torch.argmax(probs).item()

    # Sentiment labels
    labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    return labels[sentiment]

# Example usage
if __name__ == '__main__':
    test_comments = [
        'I love this product! It is amazing.',
        'This is the worst service I have ever used.',
        'It is okay, not too bad, not too good.'
    ]

    for comment in test_comments:
        sentiment = analyze_sentiment(comment)
        print(f"Comment: {comment}")
        print(f"Sentiment: {sentiment}")
        print('-' * 30)
