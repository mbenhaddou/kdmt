from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from io import BytesIO


def upload_file(connection_string, container, file_path, blob_name, overwrite=False):

    with open(file_path, "rb") as data:
        upload_file_object(connection_string, container, data, blob_name, overwrite)


def get_file_object(connect_str, container_name, blob_file):

    downloaded_buffer=get_blob(connect_str, container_name, blob_file)

    inmemoryfile = BytesIO()
    downloaded_buffer.download_to_stream(inmemoryfile)

    return inmemoryfile


def get_blob(connect_str, container_name, blob_file):

    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_file)
    streamdownloader = blob_client.download_blob()

    return streamdownloader

def upload_file_object(connect_str, container_name, file_object, blob_name, overwrite=False):




    container_client = get_container(connect_str, container_name)

    try:
        print(f"Uploading blob {blob_name}")
        container_client.upload_blob(data=file_object, name=blob_name, overwrite=overwrite)

    except Exception as e:
        raise e

    return container_client





def get_container(connection_string: str, container_name: str):
    container_client = ContainerClient.from_connection_string(
      connection_string, container_name)
    if not container_client.exists():
      container_client.create_container()
    return container_client
