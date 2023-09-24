import azure.functions as func
import sqlite3
import openai
import numpy as np
import tensorflow as tf
from pymilvus import MilvusClient
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import json
import os


labels = ["EventContext", "SpeechContext", "Speaker"]
versions = ["CONTEXT", "INTRO", "RAW SPEECH", "SUMMARY"]
zilliz_cloud_endpoint = os.getenv("ZILLIZ_CLOUD_ENDPOINT")
zilliz_cloud_access_key = os.getenv("ZILLIZ_CLOUD_ACCESS_KEY")
zilliz_cloud_container_name = os.getenv("ZILLIZ_CLOUD_CONTAINER_NAME")
openai.api_key = os.environ["OPENAI_API_KEY"]
db_path = "queries.db"
client = MilvusClient(uri=zilliz_cloud_endpoint, token=zilliz_cloud_access_key)


def search(query, label):
    embed = tf.saved_model.load(
        "C:/Users/sanjay/Desktop/Sprint 2/aze/backend/HttpTrigger/Universal Sentence Encoder/"
    )
    query_embedding = embed([query]).numpy()[0]
    res = client.query(
        collection_name=zilliz_cloud_container_name,
        filter=f'(Label == "{label}")',
        output_fields=["Label", "vector"],
    )
    print(res)
    vectors = [entry["vector"] for entry in res]
    print(vectors)
    similarities = cosine_similarity([query_embedding], vectors)
    most_similar_index = np.argmax(similarities)
    return res[most_similar_index]["Label"]


def process_with_openai(text, version):
    if version == "RAW SPEECH":
        return text
    elif version == "SUMMARY":
        prompt = f"Summarize the following text:\n{text}"
    elif version == "CONTEXT":
        prompt = f"Provide a context for the following text:\n{text}"
    else:  # Assuming INTRO as the default other option
        prompt = f"Introduce the speaker for the following text:\n{text}"

    response = openai.Completion.create(
        engine="text-davinci-002", prompt=prompt, max_tokens=150
    )
    return response.choices[0].text.strip()


def main(req: func.HttpRequest) -> func.HttpResponse:
    load_dotenv()

    try:
        # Parse the JSON body
        req_body = req.get_json()
        user_query = req_body.get("user_query")
        selected_label_index = int(req_body.get("selected_label_index"))
        selected_label = labels[selected_label_index - 1]
        version_index = int(req_body.get("version_index"))
        version = versions[version_index - 1]

        # Replace user inputs with data from HTTP request
        # user_query, selected_label, and version are now gotten from the request body

        # Rest of your logic remains largely unchanged
        answer_label = search(user_query, selected_label)
        processed_text = process_with_openai(answer_label, version)

        # Store in SQLite DB
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY,
            query TEXT,
            answer TEXT
        )
    """
        )
        cursor.execute(
            "INSERT INTO queries (query, answer) VALUES (?, ?)",
            (user_query, processed_text),
        )
        conn.commit()
        conn.close()

        # Return the processed text as the response
        return func.HttpResponse(
            json.dumps({"processed_text": processed_text}), status_code=200
        )

    except Exception as e:
        return func.HttpResponse("Error in processing request", status_code=500)
