import os
import cohere
from pdfminer.high_level import extract_text
from pymilvus import MilvusClient, connections, Collection
from pymilvus import Collection, connections
from sklearn.metrics.pairwise import cosine_similarity as cs
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import numpy as np
from openai import OpenAI
import re
import json


# Setup and API keys
cohere_api_key = os.getenv("COHERE_API_KEY")
zilliz_cloud_endpoint = os.getenv("ZILLIZ_CLOUD_ENDPOINT")
zilliz_cloud_access_key = os.getenv("ZILLIZ_CLOUD_ACCESS_KEY")
zilliz_cloud_container_name = os.getenv("ZILLIZ_CLOUD_CONTAINER_NAME")

co = cohere.Client(cohere_api_key)
openai_api = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api)
relevance_threshold = 0.1


# Functions for processing text, embedding, and storing embeddings
def read_pdf_and_chunk(file_path, chunk_size=512):
    """
    Reads a PDF file at the specified file path and divides the text into chunks of a specified size.

    Args:
        file_path (str): The path to the PDF file.
        chunk_size (int, optional): The size of each chunk in number of tokens. Defaults to 512.

    Returns:
        list: A list of text chunks, where each chunk contains a specified number of tokens from the PDF file.
    """
    print("Chunking the file")
    text = extract_text(file_path)
    tokens = word_tokenize(text)
    if not tokens:
        return []
    return [
        " ".join(tokens[i : i + chunk_size]) for i in range(0, len(tokens), chunk_size)
    ]


def write_chunks_to_file(text_chunks, chunk_ids, filename="chunks_source.txt"):
    """
    Write the text chunks and their corresponding IDs to a file in a specific format.

    Args:
        text_chunks (list): A list of text chunks.
        chunk_ids (list): A list of corresponding chunk IDs.
        filename (str, optional): The name of the file to write the chunks to. Defaults to "chunks_source.txt".

    Returns:
        None
    """
    with open(filename, "w") as file:
        for index, (chunk, chunk_id) in enumerate(zip(text_chunks, chunk_ids)):
            file.write(f"Chunk No {index + 1} (Chunk ID {chunk_id}):\n{chunk}\n\n\n")


def embed_text_chunks(chunks):
    """
    Embeds text chunks using a pre-trained model.

    Args:
        chunks (list): A list of text chunks to be embedded.

    Returns:
        list: A list of embeddings corresponding to the input text chunks.
    """
    print("Embedding Chunks")
    response = co.embed(model="large", texts=chunks)
    return response.embeddings


def store_embeddings_in_zilliz(text_chunks, embeddings, collection_name):
    """
    Store the text chunks and their corresponding embeddings in the Zilliz database.

    Args:
        text_chunks (list): A list of text chunks to be stored in the Zilliz database.
        embeddings (list): A list of embeddings corresponding to the text chunks.
        collection_name (str): The name of the collection in the Zilliz database where the data will be stored.

    Returns:
        bool: The insert result, indicating the success or failure of storing the embeddings in the Zilliz database.
    """
    milvus = MilvusClient(uri=zilliz_cloud_endpoint, token=zilliz_cloud_access_key)
    entities = []

    print("Storing embeddings in Zilliz...")
    for index, (chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
        entity = {"text": chunk, "embedding": embedding}
        entities.append(entity)

    insert_result = milvus.insert(collection_name=collection_name, data=entities)
    print("Embeddings stored in Zilliz.")
    return insert_result


def create_flag_file(flag_file="embeddings_stored.flag"):
    """
    Create an empty flag file.

    Args:
        flag_file (str, optional): The name of the flag file to be created. Defaults to "embeddings_stored.flag".

    Returns:
        None

    Example:
        create_flag_file("embeddings_stored.flag")

    This code will create an empty file named "embeddings_stored.flag" in the current directory.
    """
    with open(flag_file, "w") as file:
        file.write("")


def check_embeddings_stored(flag_file="embeddings_stored.flag"):
    return os.path.exists(flag_file)


def clean_text(text):
    return re.sub(r"[\W_]+", " ", text)


# Q&A generation, translation, and query handling
def generate_qa_pairs(chunks, chunk_ids, max_qa_pairs=50):
    """
    Generates question-answer pairs based on a list of text chunks and chunk IDs.

    Args:
        chunks (list): A list of text chunks.
        chunk_ids (list): A list of chunk IDs.
        max_qa_pairs (int, optional): The maximum number of question-answer pairs to generate. Defaults to 50.

    Returns:
        list: The generated question-answer pairs, each containing the question, answer, chunk ID, and chunk number.
    """
    print("Generating Question and Answers from:")
    qa_pairs = []

    for idx, (chunk, chunk_id) in enumerate(zip(chunks, chunk_ids)):
        if len(qa_pairs) >= max_qa_pairs:
            break

        cleaned_chunk = clean_text(chunk)
        translated_chunk = translate_text(cleaned_chunk)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"Generate a multiple set of question and answer tuples based on the following translated content make sure each question and answer has different variations:\n{translated_chunk}",
            },
        ]

        try:
            response = client.chat.completions.create(model="gpt-4", messages=messages)

            if response.choices and response.choices[0].message:
                res = response.choices[0].message.content.strip()
                qas = res.split("\n")
                for i in range(0, len(qas), 2):
                    if i + 1 < len(qas):
                        question = qas[i].replace("Q:", "").strip()
                        answer = qas[i + 1].replace("A:", "").strip()
                        if question and answer:
                            qa_pairs.append(
                                {
                                    "question": question,
                                    "answer": answer,
                                    "ChunkID": chunk_id,
                                    "Chunk_No": idx + 1,  # Sequential chunk number
                                }
                            )
                            print(f"Generated Q&A Pair {len(qa_pairs)}:")
                            print(f"Question: {question}")
                            print(f"Answer: {answer}\n")

        except OpenAI.RateLimitError:
            print(
                "API quota exceeded. Unable to generate more Q&A pairs. Please try again later."
            )
            break

        print(f"Processing chunk_{idx + 1}")

    return qa_pairs


def embed_query(query):
    """
    Generate embeddings for a given query using a language model.

    Args:
        query (str): The query for which embeddings need to be generated.

    Returns:
        list: The embeddings of the query.

    Example:
        >>> query = "What is the capital of France?"
        >>> embeddings = embed_query(query)
        >>> print(embeddings)
        [0.123, 0.456, 0.789, ...]
    """
    response = co.embed(model="large", texts=[query])
    query_embedding = response.embeddings[0]

    # Convert to NumPy array if not already and reshape to 1D
    query_embedding = np.array(query_embedding).reshape(-1)
    return query_embedding


def search_in_zilliz(query_embedding, collection_name, top_k=5):
    """
    Perform a search operation on a collection in Milvus using a query embedding.

    Args:
        query_embedding (list): A list representing the embedding of the query entity.
        collection_name (str): The name of the collection in Milvus where the search operation will be performed.
        top_k (int, optional): The number of search results to return. Default is 5.

    Returns:
        list: A list of tuples, where each tuple contains the text and ID of an entity that matches the query embedding.
              The number of tuples in the list is equal to the specified top_k parameter.
    """
    milvus = MilvusClient(uri=zilliz_cloud_endpoint, token=zilliz_cloud_access_key)
    search_params = {"metric_type": "L2", "params": {"nprobe": 16}}

    search_results = milvus.search(
        collection_name=collection_name,
        data=[query_embedding],
        search_params=search_params,
        limit=top_k,
        output_fields=["text", "id"],
    )

    return [
        (item["entity"]["text"], item["entity"]["id"])
        for sublist in search_results
        for item in sublist
    ]


def translate_text(text, source_language="it", target_language="en"):
    """
    Translates the given text from the source language to the target language using the OpenAI GPT-4 model.

    Args:
        text (str): The text to be translated.
        source_language (str, optional): The language of the input text. Defaults to "it" (Italian).
        target_language (str, optional): The language to which the text should be translated. Defaults to "en" (English).

    Returns:
        str: The translated text. If the translation is not available, returns "Translation not available".
    """
    translation_prompt = (
        f"Translate this text from {source_language} to {target_language}: {text}"
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": translation_prompt}],
    )
    return (
        response.choices[0].message.content.strip()
        if response.choices and response.choices[0].message
        else "Translation not available"
    )


# Additional functions for Q&A storage and retrieval
def generate_and_store_qa_pairs(chunks, chunk_ids, filename="qa_pairs.json"):
    """
    Generate question-answer pairs based on the given text chunks and chunk IDs, and store them in a JSON file.

    Args:
        chunks (list): A list of text chunks.
        chunk_ids (list): A list of chunk IDs.
        filename (str, optional): The name of the JSON file to store the question-answer pairs. Defaults to "qa_pairs.json".

    Returns:
        list: The generated question-answer pairs.
    """
    if os.path.exists(filename):
        print(f"Q&A file {filename} already exists. Loading existing Q&A pairs.")
        return load_qa_pairs(filename)

    qa_pairs = generate_qa_pairs(chunks, chunk_ids, max_qa_pairs=50)
    save_qa_pairs(qa_pairs, filename)
    return qa_pairs


def save_qa_pairs(qa_pairs, filename="qa_pairs.json"):
    """
    Save a list of question-answer pairs to a JSON file.

    Args:
        qa_pairs (list): A list of dictionaries representing question-answer pairs.
        filename (str, optional): The name of the JSON file to save the question-answer pairs. Defaults to "qa_pairs.json".

    Returns:
        None

    Example Usage:
        qa_pairs = [
            {"question": "What is your name?", "answer": "My name is John."},
            {"question": "How old are you?", "answer": "I am 25 years old."}
        ]
        save_qa_pairs(qa_pairs, "qa_pairs.json")

    """
    with open(filename, "w") as file:
        json.dump(qa_pairs, file)
        file.write("\n\n")


def load_qa_pairs(filename="qa_pairs.json"):
    """
    Load and return the question-answer pairs from a JSON file.

    Args:
        filename (str, optional): The name of the JSON file to load the question-answer pairs from.
                                 If not provided, the default value is "qa_pairs.json".

    Returns:
        dict: The loaded question-answer pairs as a JSON object.
    """
    with open(filename, "r") as file:
        return json.load(file)


def get_synonyms(word):
    """
    Retrieves synonyms for a given word using the WordNet corpus from the Natural Language Toolkit (NLTK) library.

    Args:
        word (str): The word for which synonyms are to be retrieved.

    Returns:
        set: A set of synonyms for the given word.
    """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms


def is_query_relevant(query, ground_truth_question):
    """
    Determines if a given query is relevant to a ground truth question by comparing the keywords in the query and the question, and expanding the keywords with synonyms.

    Args:
        query (str): The user query to be checked for relevance.
        ground_truth_question (str): The ground truth question to compare the query against.

    Returns:
        bool: A boolean value indicating whether the query is relevant to the ground truth question.
    """
    query_keywords = set(query.lower().split())
    question_keywords = set(ground_truth_question.lower().split())

    # Expand keywords with synonyms
    expanded_query_keywords = set()
    expanded_question_keywords = set()

    for word in query_keywords:
        expanded_query_keywords.update(get_synonyms(word))
    for word in question_keywords:
        expanded_question_keywords.update(get_synonyms(word))

    # Check for common keywords considering synonyms
    return len(expanded_query_keywords.intersection(expanded_question_keywords)) >= 2


def embed_text(text):
    """
    Embeds the given query using a pre-trained model and compares it with the embedded ground truth questions to calculate the cosine similarity. If the similarity is above a defined threshold, returns True indicating that the query is relevant to the ground truth questions.

    Args:
        query (str): The query to be embedded and compared with the ground truth questions.

    Returns:
        bool: A boolean value indicating whether the query is relevant to the ground truth questions.
    """
    response = co.embed(model="large", texts=[text])
    return response.embeddings[0]


def cosine_similarity(vec1, vec2):
    """
    Calculates the similarity between two sets of keywords by expanding the keywords with synonyms and checking for common keywords considering synonyms.

    Args:
        query_keywords (list): A list of keywords from the query.
        question_keywords (list): A list of keywords from the question.

    Returns:
        bool: True if the intersection of expanded query keywords and expanded question keywords has at least 2 common keywords, otherwise False.
    """
    # Ensure the vectors are NumPy arrays and flatten them to 1D
    vec1 = np.array(vec1).flatten()
    vec2 = np.array(vec2).flatten()

    # Reshape the vectors to 2D for cosine_similarity function
    vec1_reshaped = vec1.reshape(1, -1)
    vec2_reshaped = vec2.reshape(1, -1)

    # Compute and return the cosine similarity as a scalar
    similarity = cs(vec1_reshaped, vec2_reshaped)[0][0]
    return similarity


def verify_response_against_truth(generated_response, ground_truth_qas):
    """
    Embeds the given query using a pre-trained model and compares it with the embedded ground truth questions to calculate the cosine similarity. If the similarity is above a defined threshold, returns True indicating that the query is relevant to the ground truth questions.

    Args:
        query (str): The query to be embedded and compared with the ground truth questions.
        ground_truth_questions (list): A list of ground truth questions to compare the query against.

    Returns:
        bool: A boolean value indicating whether the query is relevant to the ground truth questions.
    """
    response_embedding = embed_text(generated_response)
    # Convert to NumPy array and reshape to 1D
    response_embedding = np.array(response_embedding).flatten()

    similarity_threshold = 0.7  # Define your similarity threshold

    for qa in ground_truth_qas:
        ground_truth_embedding = embed_text(qa["answer"])
        # Convert to NumPy array and reshape to 1D
        ground_truth_embedding = np.array(ground_truth_embedding).flatten()

        similarity = cosine_similarity(response_embedding, ground_truth_embedding)

        if similarity >= similarity_threshold:
            return True
    return False


def create_context_embedding(chunks):
    """
    Create a context embedding by averaging the embeddings of a list of text chunks.

    Args:
        chunks (list): A list of text chunks to be embedded.

    Returns:
        numpy.ndarray: The average embedding of the provided text chunks as a NumPy array.
                       Returns an empty array if no valid chunks are available or if embeddings are invalid.

    Note:
        This function first checks if there are any chunks to embed. If there are no chunks or embeddings are invalid,
        it returns an empty NumPy array. Otherwise, it returns the mean of all the embeddings.
    """
    if not chunks:
        print("No chunks available for embedding.")
        return np.array([])  # Return an empty array if no chunks are present

    embeddings = embed_text_chunks(chunks)

    # Check if embeddings are valid
    if not embeddings or np.isnan(embeddings).any():
        print("Invalid embeddings encountered.")
        return np.array([])  # Return an empty array if embeddings are invalid

    context_embedding = np.mean(embeddings, axis=0)
    return context_embedding


def retrieve_stored_embeddings(collection_name):
    """
    Retrieve all stored embeddings from a Milvus collection.

    Args:
        zilliz_cloud_container_name (str): The name of the Milvus collection.

    Returns:
        list: A list containing all the stored embeddings from the Milvus collection.
    """
    # Ensure connection to Milvus is established
    connections.connect(
        alias="default",
        uri=zilliz_cloud_endpoint,
        token=zilliz_cloud_access_key,
    )

    # Load the collection
    collection = Collection(name=collection_name)
    collection.load()

    # Retrieve all stored embeddings
    stored_embeddings = []
    results = collection.query(
        "id > 0", output_fields=["embedding"], limit=1000
    )  # Adjust the limit as needed
    for result in results:
        stored_embeddings.append(result["embedding"])

    return stored_embeddings


# Function to process user query and generate a response
def process_user_query(query, ground_truth_qas, collection_name, context_embedding):
    """
    Process the user query to find the most relevant chunks of text from a given context and generate a response.

    Args:
        query (str): The user query to be processed.
        ground_truth_qas (list): A list of question-answer pairs from the given context.
        zilliz_cloud_container_name (str): The name of the container where the text chunks are stored.
        context_embedding (numpy.ndarray): The embedding of the entire context.

    Returns:
        tuple: A tuple containing the retrieved chunk IDs with ranks, the generated response, and a flag indicating the response correctness.

    Raises:
        None

    Example:
        retrieved_chunks_with_rank, generated_response, is_correct = process_user_query(user_query, ground_truth_qas, zilliz_cloud_container_name, context_embedding)
    """

    query_embedding = embed_query(query)
    print("Searching relevant results with Query Embedding:")
    similarity_to_context = cosine_similarity(query_embedding, context_embedding)
    if similarity_to_context < relevance_threshold:
        print("Query is not relevant to the available data.")
        return (
            [],
            "The query does not adhere to the context of the available data.",
            False,
        )

    relevant_chunk_ids = set()
    for qa in ground_truth_qas:
        if is_query_relevant(query, qa["question"]):
            relevant_chunk_ids.add(qa["ChunkID"])

    if not relevant_chunk_ids:
        print("No relevant chunks found for the query.")
        return [], "", False

    retrieved_chunks = retrieve_relevant_chunks_by_ids(
        relevant_chunk_ids, collection_name
    )
    if not retrieved_chunks:
        print("No chunks retrieved from the database.")
        return [], "", False

    generated_response = generate_response_from_chunks(
        query, [chunk for chunk, _ in retrieved_chunks], ground_truth_qas
    )
    is_correct = verify_response_against_truth(generated_response, ground_truth_qas)
    retrieved_chunks_with_rank = [
        (chunk_id, idx + 1) for idx, (_, chunk_id) in enumerate(retrieved_chunks)
    ]

    return retrieved_chunks_with_rank, generated_response, is_correct


def get_chunk_number_mapping(filename="chunks_source.txt"):
    chunk_number_mapping = {}
    with open(filename, "r") as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith("Chunk No"):
                parts = line.split()  # Split by whitespace
                chunk_number = int(parts[2])  # Get the chunk number
                chunk_id = int(parts[5].strip("):"))  # Get the chunk ID
                chunk_number_mapping[chunk_id] = chunk_number
    return chunk_number_mapping


def retrieve_relevant_chunks_by_ids(chunk_ids, collection_name):
    """
    Retrieves chunks of text from a Milvus collection based on a set of chunk IDs.

    Args:
        chunk_ids (list): A list of chunk IDs for the chunks to be retrieved.
        collection_name (str): The name of the Milvus collection to retrieve the chunks from.

    Returns:
        list: A list of tuples containing the text and ID of the retrieved chunks.
    """
    # Ensure connection to Milvus is established
    connections.connect(
        alias="default",
        uri=zilliz_cloud_endpoint,
        token=zilliz_cloud_access_key,
    )

    # Load the collection
    collection = Collection(name=collection_name)
    collection.load()

    # Retrieve chunk number mapping from the file
    chunk_number_mapping = get_chunk_number_mapping()

    retrieved_chunks = []
    retrieved_chunk_ids = set()  # Track retrieved IDs to avoid duplication
    print("Relevant Chunk IDs:", [chunk_ids])
    for chunk_id in chunk_ids:
        if chunk_id not in retrieved_chunk_ids:
            # Construct a query expression to retrieve data by ID
            expr = f"id in [{chunk_id}]"
            fields = ["text", "id"]  # Specify the fields to retrieve

            try:
                results = collection.query(expr, output_fields=fields)
                for result in results:
                    text, id = result["text"], result["id"]
                    retrieved_chunks.append((text, id))
                    retrieved_chunk_ids.add(chunk_id)
                    chunk_number = chunk_number_mapping.get(chunk_id, "Unknown")
                    # Display each retrieved chunk in a readable format
                    print(f"Chunk {chunk_number} (ID {id}):\n{text[:200]}...\n\n")
            except Exception as e:
                print(f"Error retrieving chunk {chunk_id}: {e}")

    return retrieved_chunks


# Function to generate a response from the selected chunks and ground truth Q&A
def generate_response_from_chunks(query, chunks, ground_truth_qas):
    """
    Generate a concise and informative response to the user's query by synthesizing information from the provided chunks and ground truth Q&A.

    Args:
        query (str): The user's query.
        chunks (list of str): The relevant chunks of information.
        ground_truth_qas (list of dict): The ground truth question-answer pairs.

    Returns:
        str: The concise and informative response generated by the function.
    """
    try:
        prompt = f"Generate a concise and informative response to the user's query by synthesizing information from the provided chunks and ground truth Q&A:\n\n"
        prompt += f"User Query: '{query}'\n\n"
        prompt += "Relevant Chunks:\n"
        for idx, chunk in enumerate(chunks, start=1):
            prompt += f"Chunk {idx}: {chunk}\n\n"

        prompt += "Relevant Q&A Ground Truth:\n"
        for qa in ground_truth_qas:
            prompt += f"Q: {qa['question']}\nA: {qa['answer']}\n\n"

        prompt += "Response:"

        response = client.chat.completions.create(
            model="gpt-3.5-turbo-16k", messages=[{"role": "system", "content": prompt}]
        )

        if response.choices and response.choices[0].message:
            generated_response = response.choices[0].message.content.strip()
        else:
            generated_response = "Unable to generate a response."
    except OpenAI.RateLimitError:
        generated_response = "API quota exceeded. Please try again later."

    return generated_response


def evaluate_performance(ground_truth_qas, retrieved_chunk_ids_with_ranks):
    """
    Calculates the recall and mean reciprocal rank (MRR) for a retrieval system.

    Args:
        ground_truth_qas (list): A list of dictionaries representing the ground truth question-answer pairs.
            Each dictionary contains the keys "ChunkID", "Question", and "Answer".
        retrieved_chunk_ids_with_ranks (list): A list of tuples representing the retrieved chunk IDs and their ranks.
            Each tuple contains the chunk ID and its rank.

    Returns:
        tuple: A tuple containing the recall and mean reciprocal rank (MRR) values.

    Example:
        ground_truth_qas = [
            {"ChunkID": 1, "Question": "What is the capital of France?", "Answer": "Paris"},
            {"ChunkID": 2, "Question": "Who painted the Mona Lisa?", "Answer": "Leonardo da Vinci"},
            {"ChunkID": 3, "Question": "What is the square root of 16?", "Answer": "4"}
        ]
        retrieved_chunk_ids_with_ranks = [
            (1, 2),
            (2, 1),
            (3, 3)
        ]

        recall, mrr = evaluate_performance(ground_truth_qas, retrieved_chunk_ids_with_ranks)

        print(f"Recall: {recall}, MRR: {mrr}")

    """
    total_relevant = len(ground_truth_qas)
    retrieved_relevant = 0
    reciprocal_ranks = []

    for qa in ground_truth_qas:
        found = False
        for rank, (chunk_id, _) in enumerate(retrieved_chunk_ids_with_ranks, start=1):
            if qa["ChunkID"] == chunk_id:
                retrieved_relevant += 1
                if not found:
                    found = True
                    reciprocal_ranks.append(1.0 / rank)

    recall = retrieved_relevant / total_relevant if total_relevant > 0 else 0
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0

    return recall, mrr


def load_ground_truth_data(filename="qa_pairs.json"):
    """
    Load the ground truth data from a JSON file and return it as a Python object.

    Args:
        filename (str, optional): The name of the JSON file to be loaded. Defaults to "qa_pairs.json".

    Returns:
        object: The loaded JSON data as a Python object.
    """
    with open(filename, "r") as file:
        return json.load(file)


def calculate_map(ground_truth_qas, retrieved_chunk_ids_with_ranks):
    """
    Calculates the Mean Average Precision (MAP) score for a retrieval system.

    Args:
        ground_truth_qas (list): A list of dictionaries representing the ground truth question-answer pairs.
            Each dictionary contains the keys "ChunkID", "Question", and "Answer".
            "ChunkID" is the ID of the chunk, "Question" is the question, and "Answer" is the corresponding answer.
        retrieved_chunk_ids_with_ranks (list): A list of tuples representing the retrieved chunk IDs with their ranks.
            Each tuple contains the chunk ID and its rank.

    Returns:
        float: The Mean Average Precision (MAP) score for the retrieval system.
    """
    average_precisions = []

    for qa in ground_truth_qas:
        relevant_chunk_id = qa["ChunkID"]
        cumulative_relevant = 0
        precision_at_hits = []

        for rank, (chunk_id, _) in enumerate(retrieved_chunk_ids_with_ranks, start=1):
            if chunk_id == relevant_chunk_id:
                cumulative_relevant += 1
                precision_at_hit = cumulative_relevant / rank
                precision_at_hits.append(precision_at_hit)

        if precision_at_hits:
            average_precision = sum(precision_at_hits) / len(precision_at_hits)
            average_precisions.append(average_precision)

    map_score = np.mean(average_precisions) if average_precisions else 0
    return map_score


# Main function modifications
def main():
    """
    The main function is the main entry point of the code. It performs several tasks such as reading a PDF file, cleaning and chunking the text, embedding the text chunks, storing the embeddings, generating and storing question-answer pairs, and finally, processing user queries and finding the most relevant question-answer pair.

    Inputs:
    None

    Outputs:
    None

    Example Usage:
    main()
    """
    script_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_directory, "faq.pdf")
    flag_file = os.path.join(script_directory, "embeddings_stored.flag")
    qa_file = os.path.join(script_directory, "qa_pairs.json")

    if not os.path.exists(flag_file):
        raw_text_chunks = read_pdf_and_chunk(file_path)
        cleaned_text_chunks = [clean_text(chunk) for chunk in raw_text_chunks]
        embeddings = embed_text_chunks(cleaned_text_chunks)
        chunk_ids = store_embeddings_in_zilliz(
            cleaned_text_chunks, embeddings, zilliz_cloud_container_name
        )
        write_chunks_to_file(
            cleaned_text_chunks,
            chunk_ids,
            os.path.join(script_directory, "chunks_source.txt"),
        )
        create_flag_file(flag_file)
        ground_truth_qas = generate_and_store_qa_pairs(
            cleaned_text_chunks, chunk_ids, qa_file
        )
    else:
        print("Flag file found. Proceeding with existing data.")
        if not os.path.exists(qa_file):
            print(f"File {qa_file} does not exist.")
            return

        ground_truth_qas = load_qa_pairs(qa_file)
        stored_embeddings = retrieve_stored_embeddings(zilliz_cloud_container_name)
        if not stored_embeddings:
            print("No stored embeddings found. Exiting.")
            return
        context_embedding = np.mean(stored_embeddings, axis=0)

    while True:
        user_query = input("Enter your query (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            break
        print("Ensuring the shape of embeddings are in required shape...")
        query_embedding = embed_query(user_query)
        max_similarity = 0

        for qa in ground_truth_qas:
            ground_truth_embedding = embed_query(qa["question"])
            similarity = cosine_similarity(query_embedding, ground_truth_embedding)
            max_similarity = max(max_similarity, similarity)

        if max_similarity < relevance_threshold:
            print("Query not relevant to the context.")
            continue

        (
            retrieved_chunk_ids_with_ranks,
            generated_response,
            response_correctness,
        ) = process_user_query(
            user_query, ground_truth_qas, zilliz_cloud_container_name, context_embedding
        )

        recall, mrr = evaluate_performance(
            ground_truth_qas, retrieved_chunk_ids_with_ranks
        )
        map_score = calculate_map(ground_truth_qas, retrieved_chunk_ids_with_ranks)

        print(f"Generated Response:\n{generated_response}")
        print(f"LLM Response Correctness: {response_correctness}")
        print(f"Recall: {recall},\nMRR: {mrr},\nMAP: {map_score}")


if __name__ == "__main__":
    main()
