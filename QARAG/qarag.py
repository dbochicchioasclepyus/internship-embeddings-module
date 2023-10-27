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


def generate_qa_rag_model(topic, num_questions=10):
    """
    Generate a Question-Answer RAG (Retrieval-Augmented Generation) model based on a given topic.

    Args:
        topic (str): The topic for which the essay and questions will be generated.
        num_questions (int, optional): The number of questions to generate. Default is 10.

    Returns:
        Tuple[List[Tuple[str, str]], List[str]]: A tuple containing a list of generated question-answer pairs and a list of generated questions.
    """
    print("Generating essay based on the topic...")
    essay_prompt = f"Write an essay on the topic: {topic}."
    essay_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant in generating an essay that would help a 5 year old understand. You will not go too deep into the topic. You will provide me the surface level knowledge.",
            },
            {"role": "user", "content": essay_prompt},
        ],
        max_tokens=300,
    )
    essay = essay_response["choices"][0]["message"]["content"]
    print(essay_response)
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
    questions = [q.split(".")[1].strip() if "." in q else q.strip() for q in questions]
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


def retrieve_answers_from_zilliz(question, collection_name):
    """
    Retrieve answers from Zilliz based on the similarity to a given question.

    Args:
        question (str): The question for which answers will be retrieved.
        collection_name (str): The name of the Zilliz collection to search for answers.

    Returns:
        Tuple[str, float]: A tuple containing the closest answer and its similarity score, or (None, None) if no match is found.
    """
    milvus = MilvusClient(uri=zilliz_cloud_endpoint, token=zilliz_cloud_access_key)
    question_embedding = cohere_client.embed(
        texts=[question], model="small"
    ).embeddings[0]
    question_embedding = question_embedding / np.linalg.norm(question_embedding)
    search_params = {"metric_type": "L2", "params": {"nprobe": 16}}
    results = milvus.search(
        collection_name=collection_name,
        data=[[float(i) for i in question_embedding.tolist()]],
        params=search_params,
        top_k=1,
    )

    if results:
        closest_id = results[0][0]["id"]
        closest_distance = results[0][0]["distance"]
        closest_answer = milvus.query(
            collection_name, "id in [{}]".format(closest_id), ["answer"]
        )[0]["answer"]
        return closest_answer, closest_distance
    else:
        return None, None


def get_similarity(target, candidates):
    """
    Calculate cosine similarity scores between a target vector and a list of candidate vectors.

    Args:
        target (List[float]): The target vector.
        candidates (List[List[float]]): List of candidate vectors.

    Returns:
        List[Tuple[int, float]]: A list of tuples containing the index and similarity score of each candidate vector, sorted by similarity score in descending order.
    """
    candidates = np.array(candidates)
    target = np.expand_dims(np.array(target), axis=0)
    sim = cosine_similarity(target, candidates)
    sim = np.squeeze(sim).tolist()
    sort_index = np.argsort(sim)[::-1]
    sort_score = [sim[i] for i in sort_index]
    similarity_scores = zip(sort_index, sort_score)
    return similarity_scores


def normalize_vector(vector):
    """
    Normalize a vector to have unit length.

    Args:
        vector (List[float]): The input vector.

    Returns:
        List[float]: The normalized vector.
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector  # Avoid division by zero
    return vector / norm


def store_embeddings_in_zilliz(qa_pairs, question_embeddings, collection_name):
    """
    Store question embeddings and answers in Zilliz.

    Args:
        qa_pairs (List[Tuple[str, str]]): A list of question-answer pairs.
        question_embeddings (cohere.Embeddings): Embeddings for the questions.
        collection_name (str): The name of the Zilliz collection to store the data.

    Returns:
        Tuple[float, float]: The maximum distance for exact match and the minimum distance for barely match.
    """
    milvus = MilvusClient(uri=zilliz_cloud_endpoint, token=zilliz_cloud_access_key)
    entities = []
    max_distance_for_exact_match = 0
    min_distance_for_barely_match = float("inf")

    for qa_pair, embedding in zip(qa_pairs, question_embeddings.embeddings):
        question, answer = qa_pair
        embedding = normalize_vector(embedding)
        entity = {"qvector": embedding.tolist(), "answer": answer}
        entities.append(entity)

    print("Storing embeddings in Zilliz...")
    milvus.insert(collection_name=collection_name, data=entities)
    print("Embeddings stored in Zilliz.")

    for qa_pair, embedding in zip(qa_pairs, question_embeddings.embeddings):
        question, _ = qa_pair
        closest_answer, closest_distance = retrieve_answers_from_zilliz(
            question, collection_name
        )
        if closest_distance > max_distance_for_exact_match:
            max_distance_for_exact_match = closest_distance

        if (
            closest_distance < min_distance_for_barely_match
            and closest_answer != question
        ):
            min_distance_for_barely_match = closest_distance

    return max_distance_for_exact_match, min_distance_for_barely_match


def distance_to_similarity(distance):
    """
    Convert a distance value to a similarity score.

    Args:
        distance (float): The distance value.

    Returns:
        float: The similarity score.
    """
    return 1 / (1 + distance)


if __name__ == "__main__":
    topic = input("Enter a topic: ")
    qa_pairs, questions = generate_qa_rag_model(topic)
    question_embeddings = cohere_client.embed(texts=questions, model="small")
    (
        max_distance_for_exact_match,
        min_distance_for_barely_match,
    ) = store_embeddings_in_zilliz(
        qa_pairs, question_embeddings, zilliz_cloud_container_name
    )
    print(f"Max distance for exact match: {max_distance_for_exact_match}")
    print(f"Min distance for barely match: {min_distance_for_barely_match}")

    min_similarity_threshold = 1 / (1 + max_distance_for_exact_match)
    max_similarity_threshold = 1 / (1 + min_distance_for_barely_match)

    user_question = input("Enter a question: ")
    closest_answer, closest_distance = retrieve_answers_from_zilliz(
        user_question, zilliz_cloud_container_name
    )

    if closest_distance is not None:
        closest_similarity = 1 / (1 + closest_distance)

        if (
            min_distance_for_barely_match
            <= closest_distance
            <= max_distance_for_exact_match
            and min_similarity_threshold
            <= closest_similarity
            <= max_similarity_threshold
        ):
            print(
                f"Closest answer found: {closest_answer} (Distance: {closest_distance}, Similarity: {closest_similarity})"
            )
        else:
            print("Question out of range or out of topic.")
    else:
        print("No close match found.")
