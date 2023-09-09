import os
import json
import azure.functions as func
import tensorflow as tf
from azure.storage.blob import BlobServiceClient
from pymilvus import MilvusClient

def main(req: func.HttpRequest) -> func.HttpResponse:
    # Load sensitive information from environment variables
    connection_string = os.environ["AZURE_CONNECTION_STRING"]
    container_name = os.environ["AZURE_CONTAINER_NAME"]
    zilliz_cloud_endpoint = os.environ["ZILLIZ_CLOUD_ENDPOINT"]
    zilliz_cloud_access_key = os.environ["ZILLIZ_CLOUD_ACCESS_KEY"]
    zilliz_cloud_container_name = os.environ["ZILLIZ_CLOUD_CONTAINER_NAME"]

    try:
        req_body = req.get_json()
        blob_name = req_body.get("blob_name")

        # Create a BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)

        # Get a reference to the container
        container_client = blob_service_client.get_container_client(container_name)

        # Create a BlobClient for the chosen file
        blob_client = container_client.get_blob_client(blob_name)

        # Download the blob content as bytes
        blob_content = blob_client.download_blob().readall()

        if len(blob_content) > 40 * 1024:  # 40KB in bytes
            return func.HttpResponse("File size exceeds 40KB limit.", status_code=400)

        # Convert the bytes to a string (assuming it's a text file)
        file_content = blob_content.decode('unicode-escape')

        # Generate embeddings for the file content
        embeddings = generate_embeddings(file_content)

        # Check if any embeddings were generated
        if embeddings.numpy().size > 0 and embeddings[0] is not None:
            # Upload the embeddings to Zilliz Cloud
            upload_embeddings_to_zilliz_cloud(embeddings[0], zilliz_cloud_endpoint, zilliz_cloud_access_key, zilliz_cloud_container_name, blob_name)
        else:
            return func.HttpResponse(f"Warning: No embeddings generated for {blob_name}.", status_code=400)

        # Return a success response
        return func.HttpResponse(f"Embeddings for {blob_name} uploaded to Zilliz Cloud.", status_code=200)

    except Exception as e:
        return func.HttpResponse(f"An error occurred: {str(e)}", status_code=500)

def generate_embeddings(file_content):
    # Load the Universal Sentence Encoder model
    use_model_path = "C:/Users/sanjay/Desktop/enk/Universal Sentence Encoder/"
    embed = tf.saved_model.load(use_model_path)

    # Generate embeddings for the file content
    print("Generating embeddings...")
    embeddings = embed([file_content])

    return embeddings

def upload_embeddings_to_zilliz_cloud(embeddings, zilliz_cloud_endpoint, zilliz_cloud_access_key, zilliz_cloud_container_name, file_name):
    # Define the object name in Zilliz Cloud (e.g., using the file name)
    client = MilvusClient(
        uri=zilliz_cloud_endpoint,  # Cluster endpoint obtained from the console
        token=zilliz_cloud_access_key
    )

    # Upload the JSON data to Zilliz Cloud
    res = client.insert(
        collection_name=zilliz_cloud_container_name,
        data={
            'vector': embeddings.numpy().tolist()
        }
    )

    print(res)

    # Check if the upload was successful
    if res:
        print(f"Embeddings for {file_name} uploaded to Zilliz Cloud.")
    else:
        print(f"Failed to upload embeddings for {file_name} to Zilliz Cloud. Status code: {res.status_code}")

