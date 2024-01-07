# Import necessary libraries
# pylint: disable=maybe-no-member
# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long
from transformers import RobertaTokenizerFast,pipeline,AutoTokenizer
from flask import Flask,render_template,request,send_file
from statistics import mode
from dotenv import load_dotenv
import os
import torch
from topic_detect import get_topics
from charts import make_bubble_graph
from analyse_by_parts import analyse_by_parts

load_dotenv()
app = Flask(__name__)
tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
emotion_task_top_result = pipeline('text-classification', model='SamLowe/roberta-base-go_emotions',torch_dtype=torch.float16, device=0)
emotion_task_all_results = pipeline('text-classification', model='SamLowe/roberta-base-go_emotions', torch_dtype=torch.float16, top_k=None, device=0)
tokenizerSentiment = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
file_uploads = os.path.abspath("static/files")

@app.after_request
def add_cache_control(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = 0
    response.headers['Pragma'] = 'no-cache'
    return response

@app.route('/home',methods=["POST","GET"])
def home():
    return render_template('home.html')

@app.route('/features',methods=["POST","GET"])
def features():
    return render_template('features.html')

@app.route('/text-scanner',methods=["POST","GET"])
def text_scanner():
    if request.method == "POST":
        input = request.form["nm"]
        sentiment = classify_text_sentiment(input)
        emotion = classify_text_emotion(input)
        topics = get_topics(input)
        bubble_graphic_topics = make_bubble_graph(topics,"topics")
        texts,sentiments= analyse_by_parts(input)
        return render_template('text-scanner.html',sentiment=sentiment,emotion=emotion,input=input,bubble_graph_topics=bubble_graphic_topics,texts=texts,sentiments=sentiments)
    else:
        sentiment = "Input sentiment"
        topics = {"arts_&_culture":0,"fashion_&_style":0,"learning_&_educational":0,"science_&_technology":0,"business_&_entrepreneurs":0,"film_tv_&_video":0,"music":0,"sports":0,"celebrity_&_pop_culture":0,"fitness_&_health":0,"news_&_social_concern":0,"travel_&_adventure":0,"diaries_&_daily_life":0,"food_&_dining":0,"other_hobbies":0,"youth_&_student_life":0,"family":0,"gaming":0,"relationships":0}
        emotion = "Input emotion"
        input=""
        return render_template('text-scanner.html',sentiment=sentiment,emotion=emotion,input=input,topics=topics)

@app.route('/file-scanner',methods=["POST","GET"])
def file_scanner():
    if request.method == "POST":
        file = request.files["file"]
        file.save(os.path.join(file_uploads,file.filename))
        sentiment,emotion,topics = file_scan(file)
        bubble_graphic_topics = make_bubble_graph(topics,"topics")
        bubble_graphic_emotions = make_bubble_graph(emotion,"emotions")
        dominant_emotion_label = max(emotion, key=emotion.get)
        dominant_emotion = f" dominant emotion: {dominant_emotion_label}, score: {emotion[dominant_emotion_label]}%"
        return render_template('file-scanner.html',sentiment=sentiment,emotion=emotion,dominant_emotion=dominant_emotion,topics=topics,bubble_graph_topics=bubble_graphic_topics,bubble_graph_emotions=bubble_graphic_emotions)
    else:
        sentiment = "File sentiment"
        dominant_emotion = "Dominant file emotion"
        emotion = {'love':0,'admiration':0,'joy':0,'approval':0,'caring':0,'excitement':0,'amusement':0,'gratitude':0,'desire':0,'anger':0,'optimism':0,'disapproval':0,'grief':0,'annoyance':0,'pride':0,'curiosity':0,'neutral':0,'disgust':0,'disappointment':0,'realization':0,'fear':0,'relief':0,'confusion':0,'remorse':0,'embarrassment':0,'surprise':0,'sadness':0,'nervousness':0}
        topics = {"arts_&_culture":0,"fashion_&_style":0,"learning_&_educational":0,"science_&_technology":0,"business_&_entrepreneurs":0,"film_tv_&_video":0,"music":0,"sports":0,"celebrity_&_pop_culture":0,"fitness_&_health":0,"news_&_social_concern":0,"travel_&_adventure":0,"diaries_&_daily_life":0,"food_&_dining":0,"other_hobbies":0,"youth_&_student_life":0,"family":0,"gaming":0,"relationships":0}
    return render_template('file-scanner.html',sentiment=sentiment,emotion=emotion,dominant_emotion=dominant_emotion,topics=topics)

# Define classification function
def classify_text_sentiment(text):
    pipe = pipeline("text-classification", model="cardiffnlp/twitter-xlm-roberta-base-sentiment", torch_dtype=torch.float16,device=0)
    sentiment_result = pipe(text)[0]

    if sentiment_result['label'] == 'positive':
        result = "positive"
        sentiment = "positivity"
        return f"text sentiment: {result}, {sentiment} index: {round(sentiment_result['score']*100,2)}%"
    elif sentiment_result['label'] == 'negative':
        result = "negative"
        sentiment = "negativity"
        return f"text sentiment: {result}, {sentiment} index: {round(sentiment_result['score']*100,2)}%"
    else:
        return "text sentiment: neutral"
    
def classify_text_emotion(text):
    emotion = emotion_task_top_result(text)
    emotion = emotion[0]
    return f"most remarkable emotion: {emotion['label']}, score: {round(emotion['score']*100,2)}%"

def emotion_algorithm(emotion_list,score_list):
    most_frequent_emotion = mode(emotion_list)
    scores = []
    for element in emotion_list:
        if element == most_frequent_emotion:
            scores.append(score_list[emotion_list.index(element)])
    total_score = (sum(scores)/len(scores))*100
    return f"most remarkable emotion: {most_frequent_emotion}, score: {round(total_score,2)}%"

def scan_file_text(file,text):
    total_emotions = {'love':0,'admiration':0,'joy':0,'approval':0,'caring':0,'excitement':0,'amusement':0,'gratitude':0,'desire':0,'anger':0,'optimism':0,'disapproval':0,'grief':0,'annoyance':0,'pride':0,'curiosity':0,'neutral':0,'disgust':0,'disappointment':0,'realization':0,'fear':0,'relief':0,'confusion':0,'remorse':0,'embarrassment':0,'surprise':0,'sadness':0,'nervousness':0}
    topics = get_topics(text)
    for topic in topics:
        topics[topic] = float('%.1f'%(topics[topic]*100))
    text_emotions = emotion_task_all_results(text)[0]
    for emotion in text_emotions:
        total_emotions[emotion['label']] += emotion['score']
    for emotion in total_emotions:
        total_emotions[emotion] = float('%.1f'%(total_emotions[emotion]*100))
    sentiment = classify_text_sentiment(text,"phrase")
    file.close()
    os.remove(file.name)
    return sentiment,total_emotions,topics

def file_scan(file):
    with open(f"{file_uploads}/{file.filename}","r") as file:
        text = file.read()
        if len(text) < 800:
            sentiment, total_emotions,topics = scan_file_text(file,text)
            return sentiment, total_emotions, topics
        else:
            sentiment = "File sentiment"
            emotion = {'love':0,'admiration':0,'joy':0,'approval':0,'caring':0,'excitement':0,'amusement':0,'gratitude':0,'desire':0,'anger':0,'optimism':0,'disapproval':0,'grief':0,'annoyance':0,'pride':0,'curiosity':0,'neutral':0,'disgust':0,'disappointment':0,'realization':0,'fear':0,'relief':0,'confusion':0,'remorse':0,'embarrassment':0,'surprise':0,'sadness':0,'nervousness':0}
            topics = {"arts_&_culture":0,"fashion_&_style":0,"learning_&_educational":0,"science_&_technology":0,"business_&_entrepreneurs":0,"film_tv_&_video":0,"music":0,"sports":0,"celebrity_&_pop_culture":0,"fitness_&_health":0,"news_&_social_concern":0,"travel_&_adventure":0,"diaries_&_daily_life":0,"food_&_dining":0,"other_hobbies":0,"youth_&_student_life":0,"family":0,"gaming":0,"relationships":0}
            return  sentiment, emotion,topics
                
# Run the app
if __name__ == "__main__":
    app.run(debug=True)