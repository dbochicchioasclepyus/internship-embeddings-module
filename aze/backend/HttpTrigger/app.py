import os, json
import tensorflow as tf
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from pymilvus import (
    MilvusClient,
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

load_dotenv()

# Load sensitive information from environment variables
connection_string = os.getenv("AZURE_CONNECTION_STRING")
zilliz_cloud_endpoint = os.getenv("ZILLIZ_CLOUD_ENDPOINT")
zilliz_cloud_access_key = os.getenv("ZILLIZ_CLOUD_ACCESS_KEY")
zilliz_cloud_container_name = os.getenv("ZILLIZ_CLOUD_CONTAINER_NAME")
AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME")


def choose_blob_from_container():
    # Create a BlobServiceClient
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    # Get a reference to the container
    container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)

    # Define the allowed file extensions
    allowed_extensions = ["pdf", "txt", "json", "csv"]
    return [
        blob
        for blob in container_client.list_blob_names()
        if blob.split(".")[-1] in allowed_extensions
    ]


# Function to generate embeddings (you may need to adjust this)
def generate_embeddings(file_content):
    # Load the Universal Sentence Encoder model
    folder = "HttpTrigger/Universal Sentence Encoder"
    path = os.path.join(os.getcwd(), folder)
    embed = tf.saved_model.load(path)
    return embed([file_content])


def generate_embeddings_by_files(file_content):
    combined_text = ""
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)

    for blob_name in file_content:
        blob_client = container_client.get_blob_client(blob_name)
        blob_data = blob_client.download_blob()
        if blob_name.split(".")[1] == "pdf":
            content = str(blob_data.readall())
        else:
            content = blob_data.readall().decode("utf-8")
        combined_text += content

    print("Generating Embeddings...")
    embeddings = generate_embeddings(combined_text)
    print("Embeddings Generated Successfully")

    return embeddings.numpy().tolist()


# Function to upload embeddings to Zilliz Cloud (you may need to adjust this)
def upload_embeddings_to_zilliz_cloud(embeddings, file_name,label):
    client = MilvusClient(uri=zilliz_cloud_endpoint, token=zilliz_cloud_access_key)

    res = client.insert(
        collection_name=zilliz_cloud_container_name,
        data={"name": file_name, "vector": embeddings,"Label":label},
    )
    print("Embeddings uploaded to zilliz succesfully!!",res)

    return res


def create_zilliz_collection():
    connections.connect(
        alias="default",
        uri=zilliz_cloud_endpoint,
        secure=True,
        token=zilliz_cloud_access_key,
    )
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="Label", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=512),
    ]
    schema = CollectionSchema(fields, enable_dynamic_field=True)
    collection = Collection(name="embeddings", schema=schema)
    index_params = {"index_type": "AUTOINDEX", "metric_type": "L2", "params": {}}

    # To name the index, do as follows:
    collection.create_index(
        field_name="vector", index_params=index_params, index_name="vector_index"
    )
    collection.load()
