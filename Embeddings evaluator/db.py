from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    Index,
)
import os

# Set your Zilliz cloud endpoint and access key
zilliz_cloud_endpoint = os.getenv("ZILLIZ_CLOUD_ENDPOINT")
zilliz_cloud_access_key = os.getenv("ZILLIZ_CLOUD_ACCESS_KEY")
zilliz_cloud_container_name = os.getenv("ZILLIZ_CLOUD_CONTAINER_NAME")

# Connect to Zilliz Cloud
connections.connect(
    alias="default",
    uri=zilliz_cloud_endpoint,
    token=zilliz_cloud_access_key,
)

# Define fields
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=4096),
]

# Build the schema with auto_id enabled
schema = CollectionSchema(fields, description="embedding vectors", auto_id=True)

# Create collection
collection = Collection(name=zilliz_cloud_container_name, schema=schema)

# Define the index parameters
index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 100}}

# Create the index
index = Index(collection, "embedding", index_params)
collection.load()
