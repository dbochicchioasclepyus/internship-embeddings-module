import logging
import azure.functions as func
from pymilvus import (
    MilvusClient,
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
import openai
import cohere
from dotenv import load_dotenv
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

# Your existing code here (remove if _name_ == "_main_":)
# ...
# Load environment variables
load_dotenv()
cohere_api = os.getenv("COHERE_API_KEY")
openai_api = os.getenv("OPEN_AI_API_KEY")
zilliz_cloud_endpoint = os.getenv("ZILLIZ_CLOUD_ENDPOINT")
zilliz_cloud_access_key = os.getenv("ZILLIZ_CLOUD_ACCESS_KEY")
zilliz_cloud_container_name = os.getenv("ZILLIZ_CLOUD_CONTAINER_NAME")

# Initialize Cohere and OpenAI clients with environment variables
cohere_client = cohere.Client(cohere_api)
openai.api_key = openai_api


# Function to generate QA RAG model
def generate_qa_rag_model(topic, num_questions=10):
    print("Generating essay based on the topic...")
    essay_prompt = f"Write an essay on the topic: {topic}."
    essay_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": essay_prompt},
        ],
        max_tokens=300,
    )
    essay = essay_response["choices"][0]["message"]["content"]
    print("Essay generated.")

    qa_pairs = []

    print("Generating questions based on the essay...")
    question_prompt = (
        f"Generate {num_questions} questions based on the following essay: {essay}"
    )
    question_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant who generates questions without introduction or footer texts.",
            },
            {"role": "user", "content": question_prompt},
        ],
    )
    questions = question_response["choices"][0]["message"]["content"].split("\n")
    questions = [q.split(".")[1].strip() for q in questions]
    print("Questions generated.")

    print(f"Generating an answer for each question...")

    for i in range(0, len(questions)):
        answer_prompt = f"Answer the question:\n{questions[i]}\n based on the following essay:\n{essay}\nAnswer:"
        answer_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant who answers in not more than 2 or 3 lines.",
                },
                {"role": "user", "content": answer_prompt},
            ],
        )
        print("\nQuestion: ", questions[i])
        print("Answer: ", answer_response["choices"][0]["message"]["content"])
        answer = answer_response["choices"][0]["message"]["content"].strip()
        qa_pairs.append((questions[i], answer))

    print(f"\nAnswers generated for each question.")

    return qa_pairs, questions


# Function to retrieve closest answer from Zilliz based on similarity
def retrieve_answers_from_zilliz(question, collection_name, embeddings):
    milvus = MilvusClient(uri=zilliz_cloud_endpoint, token=zilliz_cloud_access_key)
    question_embedding = cohere_client.embed(texts=[question], model="small").embeddings
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = milvus.search(
        collection_name=collection_name,
        query={"qvector": [question_embedding]},
        params=search_params,
        top_k=1,
        data=embeddings,
    )

    if results:
        closest_id = results[0][0]["id"]
        closest_distance = results[0][0]["distance"]
        closest_answer = milvus.query(
            collection_name, "id == {}".format(closest_id), ["answer"]
        )[0]["answer"]
        print(closest_answer)
        # You can now use get_similarity to find the most similar answer, if needed.
        # similarity_scores = get_similarity(question_embedding, [embedding for each stored question])
        return closest_answer, closest_distance
    else:
        return None, None


def get_similarity(target, candidates):
    candidates = np.array(candidates)
    target = np.expand_dims(np.array(target), axis=0)
    sim = cosine_similarity(target, candidates)
    sim = np.squeeze(sim).tolist()
    sort_index = np.argsort(sim)[::-1]
    sort_score = [sim[i] for i in sort_index]
    similarity_scores = zip(sort_index, sort_score)
    return similarity_scores


# Function to store embeddings in Zilliz
def store_embeddings_in_zilliz(qa_pairs, question_embeddings):
    milvus = MilvusClient(uri=zilliz_cloud_endpoint, token=zilliz_cloud_access_key)
    collection_name = zilliz_cloud_container_name

    entities = []

    for qa_pair, embedding in zip(qa_pairs, question_embeddings):
        question, answer = qa_pair

        entity = {
            "qvector": embedding,  # Assuming embedding is already a list
            "answer": answer,
        }

        entities.append(entity)

    print("Storing embeddings in Zilliz...")
    # Insert data into the collection
    milvus.insert(collection_name=collection_name, data=entities)
    print("Embeddings stored in Zilliz.")


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Python HTTP trigger function processed a request.")

    # Extract data from the request body
    request_body = req.get_json()
    topic = request_body.get("topic")
    user_question = request_body.get("user_question")

    if not topic:
        return func.HttpResponse(
            "Please provide a topic in the request body.", status_code=400
        )

    if not user_question:
        return func.HttpResponse(
            "Please provide a user_question in the request body.", status_code=400
        )

    qa_pairs, questions = generate_qa_rag_model(topic)
    question_embeddings = cohere_client.embed(texts=questions, model="small")
    store_embeddings_in_zilliz(qa_pairs, question_embeddings)
    closest_answer, closest_distance = retrieve_answers_from_zilliz(
        user_question, zilliz_cloud_container_name, question_embeddings.embeddings
    )

    if closest_answer:
        return func.HttpResponse(
            json.dumps(
                {"closest_answer": closest_answer, "closest_distance": closest_distance}
            ),
            mimetype="application/json",
        )
    else:
        return func.HttpResponse("No close match found.", status_code=404)