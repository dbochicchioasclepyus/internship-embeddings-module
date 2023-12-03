# Embeddings Evaluator

## Description

This project is designed to process and evaluate text embeddings. It utilizes the power of AI language models and vector search engines to chunk text data, store embeddings, generate question-answer pairs, and handle user queries to find the most relevant answers.

## Features

- PDF text extraction and chunking.
- Text cleaning and preprocessing.
- Text embedding using the Cohere API.
- Storing embeddings in Zilliz Milvus for vector search.
- Q&A pair generation using OpenAI's GPT-4.
- User query processing with relevance checking.
- Performance metrics evaluation (Recall, MRR, MAP).

## How to Use

1. Set up your environment by installing the required dependencies from `requirements.txt`.
2. Place your PDF document in the project directory.
3. Run `evt.py` to start the process.
4. Interact with the program via the console to input queries and receive responses.

## Configuration

Before running the script, ensure you have set up the following environment variables:

- `COHERE_API_KEY`: Your Cohere API key for embeddings.
- `ZILLIZ_CLOUD_ENDPOINT`: Endpoint for Zilliz Milvus instance.
- `ZILLIZ_CLOUD_ACCESS_KEY`: Access key for Zilliz Cloud.
- `ZILLIZ_CLOUD_CONTAINER_NAME`: Container name in Zilliz Cloud.
- `OPENAI_API_KEY`: Your OpenAI API key for GPT-4.

## Documentation

Detailed docstrings are provided within the code for each function, explaining their purpose, arguments, and usage.

## Contributions

Contributions are welcome. Please open an issue to discuss the changes or submit a pull request.

## License

[MIT](LICENSE.md)
