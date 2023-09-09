import os
import json
import requests
import tensorflow as tf
from azure.storage.blob import BlobServiceClient, ContainerClient
from dotenv import load_dotenv
from pymilvus import MilvusClient
# Load environment variables from .env file
load_dotenv()

# Load sensitive information from environment variables
connection_string = os.getenv("AZURE_CONNECTION_STRING")
container_name = os.getenv("AZURE_CONTAINER_NAME")
zilliz_cloud_endpoint = os.getenv("ZILLIZ_CLOUD_ENDPOINT")
zilliz_cloud_access_key = os.getenv("ZILLIZ_CLOUD_ACCESS_KEY")
zilliz_cloud_container_name = os.getenv("ZILLIZ_CLOUD_CONTAINER_NAME")

def choose_blob_from_container(connection_string, container_name):
    # Create a BlobServiceClient
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    # Get a reference to the container
    container_client = blob_service_client.get_container_client(container_name)

    # Define the allowed file extensions
    allowed_extensions = ['.pdf', '.txt', '.json', '.csv']

    # List all the blobs in the container with allowed extensions
    blobs = [blob for blob in container_client.list_blobs() if os.path.splitext(blob.name)[1].lower() in allowed_extensions]
    
    # Display the blobs for the user to choose
    for i, blob in enumerate(blobs):
        print(f"{i + 1}. {blob.name}")

    # Ask the user to choose a blob
    choice = int(input("Enter the number of the blob you want to choose: ")) - 1

    # Return the chosen blob's name
    return blobs[choice].name

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
    uri=zilliz_cloud_endpoint, # Cluster endpoint obtained from the console
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

# Let the user choose a blob
chosen_blob_name = choose_blob_from_container(connection_string, container_name)

# Create a BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Get a reference to the container
container_client = blob_service_client.get_container_client(container_name)

# Create a BlobClient for the chosen file
blob_client = container_client.get_blob_client(chosen_blob_name)

# Download the blob content as bytes
blob_content = blob_client.download_blob().readall()

if len(blob_content) > 40 * 1024:  # 40KB in bytes
    print("File size exceeds 40KB limit. Terminating process. Please select a File under the size limit (i.e., 40KB)")
else:
    # Convert the bytes to a string (assuming it's a text file)
    file_content = blob_content.decode('unicode-escape')

    # Generate embeddings for the file content
    embeddings = generate_embeddings(file_content)

    # Check if any embeddings were generated
    if embeddings.numpy().size > 0 and embeddings[0] is not None:
        # Upload the embeddings to Zilliz Cloud
        upload_embeddings_to_zilliz_cloud(embeddings[0], zilliz_cloud_endpoint, zilliz_cloud_access_key, zilliz_cloud_container_name, chosen_blob_name)
    else:
        print(f"Warning: No embeddings generated for {chosen_blob_name}.")

# Print a message indicating that the embedding process is completed
print("Embedding process completed.")
