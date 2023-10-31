import json
import azure.functions as func
from .app import *


def main(req: func.HttpRequest) -> func.HttpResponse:
    route = req.params.get("action")
    if req.method == "GET" and route == "list":
        blobs = choose_blob_from_container()
        res = [{"name": i} for i in blobs]
        return func.HttpResponse(json.dumps(res), status_code=200)

    elif req.method == "POST" and route == "embed":
        try:
            req_body = req.get_body().decode("utf-8")
            request_data = json.loads(req_body)

            embeddings = generate_embeddings_by_files(request_data)

            response_data = {
                "message": "Embeddings generated successfully",
                "data": embeddings,
                "name": request_data,
            }
            print(request_data)
            return func.HttpResponse(json.dumps(response_data), status_code=200)
        except Exception as e:
            print(e)
            return func.HttpResponse(f"Error: {str(e)}", status_code=400)
    elif req.method == "POST" and route == "upload":
        try:
            req_body = req.get_body().decode("utf-8")
            embeddings = json.loads(req_body)["data"]
            file_name = json.loads(req_body)["name"]
            label = json.loads(req_body)["label"]
            if len(embeddings[0]) > 0:
                result = ""
                if result := upload_embeddings_to_zilliz_cloud(
                    embeddings[0], str(file_name), label
                ):
                    response_data = {
                        "message": f"Embeddings for {file_name} uploaded to Zilliz Cloud.",
                    }
                    return func.HttpResponse(json.dumps(response_data), status_code=200)
                else:
                    response_data = {
                        "message": f"Failed to upload embeddings for {file_name} to Zilliz Cloud.",
                    }
                    return func.HttpResponse(
                        json.dumps(response_data),
                        status_code=400,
                    )
            else:
                return func.HttpResponse(
                    "Warning: No embeddings in the request.", status_code=400
                )
        except Exception as e:
            print(e)
            return func.HttpResponse(f"Error: {str(e)}", status_code=400)
    else:
        return func.HttpResponse("Invalid action or HTTP method", status_code=400)
