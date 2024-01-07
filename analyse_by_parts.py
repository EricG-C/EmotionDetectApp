from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification,pipeline


sentiment_task = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment", tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment")
def analyse_by_parts(text):
    texts = text.split(". ")
    sentiments = []
    for phrase in texts:
        sentiment = sentiment_task(phrase)[0]['label']
        sentiments.append(sentiment)
    return texts, sentiments
