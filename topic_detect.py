from transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification,pipeline
from transformers import AutoTokenizer

MODEL = "cardiffnlp/tweet-topic-21-multi"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
topic_task = pipeline("text-classification", model=MODEL, tokenizer=tokenizer,top_k=None,device=0)

def get_topics(text):

    topics = {"arts_&_culture":0,"fashion_&_style":0,"learning_&_educational":0,"science_&_technology":0,"business_&_entrepreneurs":0,"film_tv_&_video":0,"music":0,"sports":0,"celebrity_&_pop_culture":0,"fitness_&_health":0,"news_&_social_concern":0,"travel_&_adventure":0,"diaries_&_daily_life":0,"food_&_dining":0,"other_hobbies":0,"youth_&_student_life":0,"family":0,"gaming":0,"relationships":0}
    output = topic_task(text)[0]

    for dict in output:
        topics[dict['label']] += dict['score']
    return topics

