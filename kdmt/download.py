import requests
import zipfile
import os, re
from pathlib import Path
from tqdm import tqdm
from typing import Union, Optional, List
from urllib.parse import  urlparse
from hashlib import md5
from kdmt.zip import untar, ungzip
import shutil

def download_decompress(url: str,
                        download_path: Union[Path, str],
                        extract_paths: Optional[Union[List[Union[Path, str]], Path, str]] = None,
                        headers: Optional[dict] = None) -> None:
    """Download and extract .tar.gz or .gz file to one or several target locations.

    The archive is deleted if extraction was successful.

    Args:
        url: URL for file downloading.
        download_path: Path to the directory where downloaded file will be stored until the end of extraction.
        extract_paths: Path or list of paths where contents of archive will be extracted.
        headers: Headers for file server.

    """
    file_name = Path(urlparse(url).path).name
    download_path = Path(download_path)

    if extract_paths is None:
        extract_paths = [download_path]
    elif isinstance(extract_paths, list):
        extract_paths = [Path(path) for path in extract_paths]
    else:
        extract_paths = [Path(extract_paths)]

    cache_dir = os.getenv('DP_CACHE_DIR')
    extracted = False
    if cache_dir:
        cache_dir = Path(cache_dir)
        url_hash = md5(url.encode('utf8')).hexdigest()[:15]
        arch_file_path = cache_dir / url_hash
        extracted_path = cache_dir / (url_hash + '_extracted')
        extracted = extracted_path.exists()
        if not extracted and not arch_file_path.exists():
            download(url, arch_file_path, headers)
        else:
            if extracted:
                print(f'Found cached and extracted {url} in {extracted_path}')
            else:
                print(f'Found cached {url} in {arch_file_path}')
    else:
        arch_file_path = download_path / file_name
        download(url, arch_file_path, headers)
        extracted_path = extract_paths.pop()

    if not extracted:
        print('Extracting {} archive into {}'.format(arch_file_path, extracted_path))
        extracted_path.mkdir(parents=True, exist_ok=True)

        if file_name.endswith('.tar.gz'):
            untar(arch_file_path, extracted_path)
        elif file_name.endswith('.gz'):
            ungzip(arch_file_path, extracted_path / Path(file_name).with_suffix('').name)
        elif file_name.endswith('.zip'):
            with zipfile.ZipFile(arch_file_path, 'r') as zip_ref:
                zip_ref.extractall(extracted_path)
        else:
            raise RuntimeError(f'Trying to extract an unknown type of archive {file_name}')

        if not cache_dir:
            arch_file_path.unlink()

    for extract_path in extract_paths:
        for src in extracted_path.iterdir():
            dest = extract_path / src.name
            if src.is_dir():
                _copytree(src, dest)
            else:
                extract_path.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(src), str(dest))



def _copytree(src: Path, dest: Path) -> None:
    """Recursively copies directory.

    Destination directory could exist (unlike if we used shutil.copytree).

    Args:
        src: Path to copied directory.
        dest: Path to destination directory.

    """
    dest.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        f_dest = dest / f.name
        if f.is_dir():
            _copytree(f, f_dest)
        else:
            shutil.copy(str(f), str(f_dest))

def download(url: str, destination: Union[Path, str], headers: Optional[dict] = None) -> None:
    """Download a file from URL to target location.

    Displays a progress bar to the terminal during the download process.

    Args:
        url: The source URL.
        destination: Path to the file destination (including file name).
        headers: Headers for file server.

    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)


    if url.startswith('s3://'):
        return s3_download(url, str(destination))

    chunk_size = 32 * 1024
    temporary = destination.with_suffix(destination.suffix + '.part')

    r = requests.get(url, stream=True, headers=headers)
    if r.status_code != 200:
        raise RuntimeError(f'Got status code {r.status_code} when trying to download {url}')
    total_length = int(r.headers.get('content-length', 0))

    if temporary.exists() and temporary.stat().st_size > total_length:
        temporary.write_bytes(b'')  # clearing temporary file when total_length is inconsistent

    with temporary.open('ab') as f:
        done = False
        downloaded = f.tell()
        if downloaded != 0:
            print(f'Warning: Found a partial download {temporary}')
        with tqdm(initial=downloaded, total=total_length, unit='B', unit_scale=True) as pbar:
            while not done:
                if downloaded != 0:
                    print(f'Warning: Download stopped abruptly, trying to resume from {downloaded} '
                                f'to reach {total_length}')
                    headers['Range'] = f'bytes={downloaded}-'
                    r = requests.get(url, headers=headers, stream=True)
                    if 'content-length' not in r.headers or \
                            total_length - downloaded != int(r.headers['content-length']):
                        raise RuntimeError(f'It looks like the server does not support resuming '
                                           f'downloads.')
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:  # filter out keep-alive new chunks
                        downloaded += len(chunk)
                        pbar.update(len(chunk))
                        f.write(chunk)
                if downloaded >= total_length:
                    # Note that total_length is 0 if the server didn't return the content length,
                    # in this case we perform just one iteration and assume that we are done.
                    done = True

    temporary.rename(destination)


def s3_download(url: str, destination: str) -> None:
    """Download a file from an Amazon S3 path `s3://<bucket_name>/<key>`

    Requires the boto3 library to be installed and AWS credentials being set
    via environment variables or a credentials file

    Args:
        url: The source URL.
        destination: Path to the file destination (including file name).
    """
    import boto3

    s3 = boto3.resource('s3', endpoint_url=os.environ.get('AWS_ENDPOINT_URL'))

    bucket, key = url[5:].split('/', maxsplit=1)
    file_object = s3.Object(bucket, key)
    file_size = file_object.content_length
    with tqdm(total=file_size, unit='B', unit_scale=True) as pbar:
        file_object.download_file(destination, Callback=pbar.update)


# def default_downloader(directory, urls, filenames, url_prefix=None,
#                        clear=False):
#     """Downloads or clears files from URLs and filenames.
#
#     Parameters
#     ----------
#     directory : str
#         The directory in which downloaded files are saved.
#     urls : list
#         A list of URLs to download.
#     filenames : list
#         A list of file names for the corresponding URLs.
#     url_prefix : str, optional
#         If provided, this is prepended to filenames that
#         lack a corresponding URL.
#     clear : bool, optional
#         If `True`, delete the given filenames from the given
#         directory rather than download them.
#
#     """
#     # Parse file names from URL if not provided
#     for i, url in enumerate(urls):
#         filename = filenames[i]
#         if not filename:
#             filename = filename_from_url(url)
#         if not filename:
#             raise ValueError("no filename available for URL '{}'".format(url))
#         filenames[i] = filename
#     files = [os.path.join(directory, f) for f in filenames]
#
#     if clear:
#         for f in files:
#             if os.path.isfile(f):
#                 os.remove(f)
#     else:
#         print('Downloading ' + ', '.join(filenames) + '\n')
#         ensure_directory_exists(directory)
#
#         for url, f, n in zip(urls, files, filenames):
#             if not url:
#                 if url_prefix is None:
#                     raise Exception("URL is not provided for file")
#                 url = url_prefix + n
#             with open(f, 'wb') as file_handle:
#                 download(url, file_handle)
#
#
# def download_from_url(url, path=None, overwrite=False, hash_value=None,
#                       hash_type="sha256", logger=None):
#     """Download file, with logic (from tensor2tensor) for Google Drive. Returns
#     the path to the downloaded file.
#
#     Args:
#         url: the url of the file from URL header. (None)
#         root: download folder used to store the file in (.data)
#         overwrite: overwrite existing files (False)
#         hash_value (str, optional): hash for url (Default: ``None``).
#         hash_type (str, optional): hash type, among "sha256" and "md5" (Default: ``"sha256"``).
#
#     Examples:
#         >>> import kdmt
#         >>> url = 'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz'
#         >>> kdmt.download.download_from_url(url)
#         >>> url = 'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz'
#         >>> kdmt.download.download_from_url(url)
#         >>> '.data/validation.tar.gz'
#
#     """
#     if path is not None:
#         path = os.path.abspath(path)
#
#
#     def _check_hash(path):
#         if hash_value:
#             if logger:
#                 logger.info('Validating hash {} matches hash of {}'.format(hash_value, path))
#             with open(path, "rb") as file_obj:
#                 if not validate_file_hash(file_obj, hash_value, hash_type):
#                     raise RuntimeError("The hash of {} does not match. Delete the file manually and retry.".format(os.path.abspath(path)))
#
#     def _process_response(r, root, filename):
#         chunk_size = 16 * 1024
#         total_size = int(r.headers.get('Content-length', 0))
#         if filename is None:
#             if 'content-disposition' not in r.headers:
#                 raise RuntimeError("Internal error: headers don't contain content-disposition.")
#             d = r.headers['content-disposition']
#             filename = re.findall("filename=\"(.+)\"", d)
#             if filename is None:
#                 raise RuntimeError("Filename could not be autodetected")
#             filename = filename[0]
#         path = os.path.join(root, filename)
#         if os.path.exists(path):
#             if logger:
#                 logger.info('File %s already exists.' % path)
#             if not overwrite:
#                 _check_hash(path)
#                 return path
#             if logger:
#                 logger.info('Overwriting file %s.' % path)
#         if logger:
#             logger.info('Downloading file {} to {}.'.format(filename, path))
#         with open(path, "wb") as file:
#             with tqdm(total=total_size, unit='B',
#                       unit_scale=1, desc=path.split('/')[-1]) as t:
#                 for chunk in r.iter_content(chunk_size):
#                     if chunk:
#                         file.write(chunk)
#                         t.update(len(chunk))
#         if logger:
#             logger.info('File {} downloaded.'.format(path))
#
#         _check_hash(path)
#         return path
#
#     if path is None:
#         _, filename = os.path.split(url)
#     else:
#         root, filename = os.path.split(os.path.abspath(path))
#
#     # skip requests.get if path exists and not overwrite.
#     if os.path.exists(path):
#         if logger:
#             logger.info('File %s already exists.' % path)
#         if not overwrite:
#             _check_hash(path)
#             return path
#
#     if 'drive.google.com' not in url:
#         response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, stream=True)
#         return _process_response(response, root, filename)
#     else:
#         # google drive links get filename from google drive
#         filename = None
#
#     if logger:
#         logger.info('Downloading from Google Drive; may take a few minutes')
#     confirm_token = None
#     session = requests.Session()
#     response = session.get(url, stream=True)
#     for k, v in response.cookies.items():
#         if k.startswith("download_warning"):
#             confirm_token = v
#     if confirm_token is None:
#         if "Quota exceeded" in str(response.content):
#             raise RuntimeError(
#                 "Google drive link {} is currently unavailable, because the quota was exceeded.".format(
#                     url
#                 ))
#         else:
#             raise RuntimeError("Internal error: confirm_token was not found in Google drive link.")
#
#     if confirm_token:
#         url = url + "&confirm=" + confirm_token
#         response = session.get(url, stream=True)
#
#     return _process_response(response, root, filename)

