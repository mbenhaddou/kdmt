from  logging import getLogger as get_logger

def create_bucket(project_name: str, bucket_name: str):
    """
    Creates a bucket on Google Cloud Platform if it does not exists already

    Example
    -------
    >>> _create_bucket_gcp(project_name='GCP-Essentials', bucket_name='test-pycaret-gcp')

    Parameters
    ----------
    project_name : str
        A Project name on GCP Platform (Must have been created from console).

    bucket_name : str
        Name of the storage bucket to be created if does not exists already.

    Returns
    -------
    None
    """
    logger=get_logger()
    # bucket_name = "your-new-bucket-name"
    import google.auth.exceptions
    from google.cloud import storage

    try:
        storage_client = storage.Client(project_name)
    except google.auth.exceptions.DefaultCredentialsError:
        logger.error(
            "Environment variable GOOGLE_APPLICATION_CREDENTIALS not set. For more information,"
            " please see https://cloud.google.com/docs/authentication/getting-started"
        )
        raise ValueError(
            "Environment variable GOOGLE_APPLICATION_CREDENTIALS not set. For more information,"
            " please see https://cloud.google.com/docs/authentication/getting-started"
        )

    buckets = storage_client.list_buckets()

    if bucket_name not in buckets:
        bucket = storage_client.create_bucket(bucket_name)
        logger.info("Bucket {} created".format(bucket.name))
    else:
        raise FileExistsError("{} already exists".format(bucket_name))


def upload_blob(
    project_name: str,
    bucket_name: str,
    source_file_name: str,
    destination_blob_name: str,
):
    """
    Upload blob to GCP storage bucket

    Example
    -------
    >>> _upload_blob_gcp(project_name='GCP-Essentials', bucket_name='test-pycaret-gcp', \
                        source_file_name='model-101.pkl', destination_blob_name='model-101.pkl')

    Parameters
    ----------
    project_name : str
        A Project name on GCP Platform (Must have been created from console).

    bucket_name : str
        Name of the storage bucket to be created if does not exists already.

    source_file_name : str
        A blob/file name to copy to GCP

    destination_blob_name : str
        Name of the destination file to be stored on GCP

    Returns
    -------
    None
    """

    logger = get_logger()

    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"
    import google.auth.exceptions
    from google.cloud import storage

    try:
        storage_client = storage.Client(project_name)
    except google.auth.exceptions.DefaultCredentialsError:
        logger.error(
            "Environment variable GOOGLE_APPLICATION_CREDENTIALS not set. For more information,"
            " please see https://cloud.google.com/docs/authentication/getting-started"
        )
        raise ValueError(
            "Environment variable GOOGLE_APPLICATION_CREDENTIALS not set. For more information,"
            " please see https://cloud.google.com/docs/authentication/getting-started"
        )

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    logger.info(
        "File {} uploaded to {}.".format(source_file_name, destination_blob_name)
    )


def _download_blob(
    project_name: str,
    bucket_name: str,
    source_blob_name: str,
    destination_file_name: str,
):
    """
    Download a blob from GCP storage bucket

    Example
    -------
    >>> _download_blob_gcp(project_name='GCP-Essentials', bucket_name='test-pycaret-gcp', \
                          source_blob_name='model-101.pkl', destination_file_name='model-101.pkl')

    Parameters
    ----------
    project_name : str
        A Project name on GCP Platform (Must have been created from console).

    bucket_name : str
        Name of the storage bucket to be created if does not exists already.

    source_blob_name : str
        A blob/file name to download from GCP bucket

    destination_file_name : str
        Name of the destination file to be stored locally

    Returns
    -------
    Model Object
    """

    logger = get_logger()

    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"
    import google.auth.exceptions
    from google.cloud import storage

    try:
        storage_client = storage.Client(project_name)
    except google.auth.exceptions.DefaultCredentialsError:
        logger.error(
            "Environment variable GOOGLE_APPLICATION_CREDENTIALS not set. For more information,"
            " please see https://cloud.google.com/docs/authentication/getting-started"
        )
        raise ValueError(
            "Environment variable GOOGLE_APPLICATION_CREDENTIALS not set. For more information,"
            " please see https://cloud.google.com/docs/authentication/getting-started"
        )

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    if destination_file_name is not None:
        blob.download_to_filename(destination_file_name)

        logger.info(
            "Blob {} downloaded to {}.".format(source_blob_name, destination_file_name)
        )

    return blob
