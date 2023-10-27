# Question-Answer RAG (Retrieval-Augmented Generation) Model

This project demonstrates the creation of a Question-Answer RAG model using OpenAI's GPT-3.5 Turbo and Cohere for generating questions based on an essay, storing embeddings in Zilliz, and retrieving answers based on similarity.

## Requirements

- Python 3.7 or higher
- Install required libraries using `pip install -r requirements.txt`
- Environment variables for API keys and Zilliz Cloud configuration (see below)

## Configuration

Before running the code, make sure to set the following environment variables:

- `COHERE_API_KEY`: Your Cohere API key.
- `OPEN_AI_API_KEY`: Your OpenAI API key.
- `ZILLIZ_CLOUD_ENDPOINT`: The endpoint URL for Zilliz Cloud.
- `ZILLIZ_CLOUD_ACCESS_KEY`: Your Zilliz Cloud access key.
- `ZILLIZ_CLOUD_CONTAINER_NAME`: The name of the Zilliz Cloud container for storing embeddings.

## Usage

1. Run the main script `qa_rag_model.py`.

2. Enter a topic when prompted. The script will generate an essay and a set of questions based on the topic.

3. The generated questions will be stored in Zilliz along with their embeddings.

4. You can then enter a question, and the script will retrieve the closest answer from Zilliz based on similarity.

## Functions

### `generate_qa_rag_model(topic, num_questions=10)`

Generate a Question-Answer RAG model based on a given topic.

- `topic` (str): The topic for which the essay and questions will be generated.
- `num_questions` (int, optional): The number of questions to generate. Default is 10.

Returns:

- A list of generated question-answer pairs.
- A list of generated questions.

### `retrieve_answers_from_zilliz(question, collection_name)`

Retrieve answers from Zilliz based on the similarity to a given question.

- `question` (str): The question for which answers will be retrieved.
- `collection_name` (str): The name of the Zilliz collection to search for answers.

Returns:

- A tuple containing the closest answer and its similarity score, or (None, None) if no match is found.

### `get_similarity(target, candidates)`

Calculate cosine similarity scores between a target vector and a list of candidate vectors.

- `target` (List[float]): The target vector.
- `candidates` (List[List[float]]): List of candidate vectors.

Returns:

- A list of tuples containing the index and similarity score of each candidate vector, sorted by similarity score in descending order.

### `normalize_vector(vector)`

Normalize a vector to have unit length.

- `vector` (List[float]): The input vector.

Returns:

- The normalized vector.

### `store_embeddings_in_zilliz(qa_pairs, question_embeddings, collection_name)`

Store question embeddings and answers in Zilliz.

- `qa_pairs` (List[Tuple[str, str]]): A list of question-answer pairs.
- `question_embeddings` (cohere.Embeddings): Embeddings for the questions.
- `collection_name` (str): The name of the Zilliz collection to store the data.

Returns:

- The maximum distance for exact match and the minimum distance for barely match.

### `distance_to_similarity(distance)`

Convert a distance value to a similarity score.

- `distance` (float): The distance value.

Returns:

- The similarity score.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
