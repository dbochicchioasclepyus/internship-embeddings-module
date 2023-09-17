import sqlite3
import os
import openai
import requests
import azure.functions as func
from dotenv import load_dotenv

# Configuration details
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]
ZILLIZ_CLOUD_ENDPOINT = os.environ["ZILLIZ_CLOUD_ENDPOINT"]
ZILLIZ_CLOUD_ACCESS_KEY = os.environ["ZILLIZ_CLOUD_ACCESS_KEY"]
ZILLIZ_CLOUD_CONTAINER_NAME = os.environ["ZILLIZ_CLOUD_CONTAINER_NAME"]
DATABASE_NAME = "text_data.db"

# Database initialization
def init_db():
    with sqlite3.connect(DATABASE_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS generated_texts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                context TEXT,
                intro TEXT,
                raw_speech TEXT,
                summary TEXT
            )
        ''')
        conn.commit()

init_db()

def main(req: func.HttpRequest) -> func.HttpResponse:
    feedback = []

    try:
        embeddings = get_embeddings_from_zilliz()
        feedback.append("Successfully fetched embeddings from Zilliz Cloud.")
        print(embeddings)
    except ValueError as e:
        return func.HttpResponse(str(e), status_code=500)

    #transcript = req.get_json().get('transcript', '')
    transcript = ""

    event_context = generate_text_from_embedding(embeddings["EventContext"], "Provide general details about the event")
    feedback.append("Generated event context successfully.")

    intro = generate_text_from_embedding(embeddings["Speaker"] + embeddings["SpeechContext"], "Provide a brief introduction about the speaker and the topic of their speech")
    feedback.append("Generated speaker introduction successfully.")

    raw_speech = extract_raw_speech_from_transcript(transcript) or generate_text_from_embedding(embeddings["SpeechContext"], "Generate a fictional speech based on the context")
    feedback.append("Extracted/Generated raw speech successfully.")

    summary = summarize_text(raw_speech)
    feedback.append("Generated speech summary successfully.")

    store_in_database(event_context, intro, raw_speech, summary)
    feedback.append("Stored generated texts in database.")

    return func.HttpResponse("\n".join(feedback), status_code=200)

def get_embeddings_from_zilliz():
    headers = {
        "Authorization": f"Bearer {ZILLIZ_CLOUD_ACCESS_KEY}"
    }
    json_data = {
    'collectionName': ZILLIZ_CLOUD_CONTAINER_NAME,
    "id": "444018636918706219"
}
    
    response = requests.post(f'{ZILLIZ_CLOUD_ENDPOINT}/v1/vector/get', headers=headers,json=json_data)
    if response.status_code != 200:
        raise ValueError("Failed to fetch embeddings from Zilliz cloud.")
    
    return response.json()

def generate_text_from_embedding(embedding, prompt):
    response = openai.Completion.create(engine="text-davinci-004", prompt=prompt, max_tokens=100)
    return response.choices[0].text.strip()

def extract_raw_speech_from_transcript(transcript):
    return transcript

def summarize_text(text):
    response = openai.Completion.create(engine="text-davinci-004", prompt=f"Summarize the following speech: {text}", max_tokens=int(len(text)/2))
    return response.choices[0].text.strip()

def store_in_database(context, intro, raw_speech, summary):
    with sqlite3.connect(DATABASE_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO generated_texts (context, intro, raw_speech, summary)
            VALUES (?, ?, ?, ?)
        ''', (context, intro, raw_speech, summary))
        conn.commit()
