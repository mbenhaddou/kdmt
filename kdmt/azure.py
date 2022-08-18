from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from io import BytesIO
import base64
import dill

def upload_file(connect_str, container_name, local_file, blob_name):

    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    container = ContainerClient.from_connection_string(connect_str, blob_name)

    try:
        container_properties = container.get_container_properties()
        # Container foo exists. You can now use it.

    except Exception as e:
        # Container foo does not exist. You can now create it.
        container_client = blob_service_client.create_container(container_name)
        # Create a blob client using the local file name as the name for the blob
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    print("\nUploading to Azure Storage as blob")

    # Upload the created file
    try:
        with open(local_file, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
    except Exception as e:
        raise e

    return container_client

def get_file(connect_str, container_name, blob_file):

    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    try:
        blob_client = blob_service_client.get_container_client(container=container_name)
    except Exception as e:
        raise e

    downloaded_buffer = blob_client.download_blob(blob_file).readall()

    inmemoryfile = BytesIO(downloaded_buffer)

    return inmemoryfile


def get_blob(connect_str, container_name, blob_file):

    try:
        blob_client = BlobClient.from_connection_string(connect_str, container_name, blob_file)
    except Exception as e:
        raise e

    downloaded_buffer = blob_client.download_blob(0)


    return downloaded_buffer

def upload_file_object(connect_str, container_name, file_object, blob_name, convert_to_base64):

    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    container_client=None
    if not blob_client:
        container_client = blob_service_client.create_container(container_name)
        # Create a blob client using the local file name as the name for the blob
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    print("\nUploading to Azure Storage as blob")

    # Upload the created file
    try:
        blob_client.upload_blob(Base64Converter(file_object), overwrite=True)
    except Exception as e:
        raise e

    return container_client

def Base64Converter(ObjectFile):
    bytes_container = BytesIO()
    dill.dump(ObjectFile, bytes_container)
    bytes_container.seek(0)
    bytes_file = bytes_container.read()
    base64File = base64.b64encode(bytes_file)
    return base64File
