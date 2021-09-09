import requests
from progressbar import UnknownLength
from kdmt.progress import progress_bar
from kdmt.file import ensure_directory_exists, filename_from_url, validate_file_hash
import os, re
from tqdm import tqdm

def download(url, file_handle, chunk_size=1024):
    """Downloads a given URL to a specific file.

    Parameters
    ----------
    url : str
        URL to download.
    file_handle : file
        Where to save the downloaded URL.

    """
    r = requests.get(url, stream=True)
    total_length = r.headers.get('content-length')
    if total_length is None:
        maxval = UnknownLength
    else:
        maxval = int(total_length)
    name = file_handle.name
    with progress_bar(name=name, maxval=maxval) as bar:
        for i, chunk in enumerate(r.iter_content(chunk_size)):
            if total_length:
                bar.update(i * chunk_size)
            file_handle.write(chunk)


def default_downloader(directory, urls, filenames, url_prefix=None,
                       clear=False):
    """Downloads or clears files from URLs and filenames.

    Parameters
    ----------
    directory : str
        The directory in which downloaded files are saved.
    urls : list
        A list of URLs to download.
    filenames : list
        A list of file names for the corresponding URLs.
    url_prefix : str, optional
        If provided, this is prepended to filenames that
        lack a corresponding URL.
    clear : bool, optional
        If `True`, delete the given filenames from the given
        directory rather than download them.

    """
    # Parse file names from URL if not provided
    for i, url in enumerate(urls):
        filename = filenames[i]
        if not filename:
            filename = filename_from_url(url)
        if not filename:
            raise ValueError("no filename available for URL '{}'".format(url))
        filenames[i] = filename
    files = [os.path.join(directory, f) for f in filenames]

    if clear:
        for f in files:
            if os.path.isfile(f):
                os.remove(f)
    else:
        print('Downloading ' + ', '.join(filenames) + '\n')
        ensure_directory_exists(directory)

        for url, f, n in zip(urls, files, filenames):
            if not url:
                if url_prefix is None:
                    raise Exception("URL is not provided for file")
                url = url_prefix + n
            with open(f, 'wb') as file_handle:
                download(url, file_handle)



def download_from_url(url, path=None, root='.data', overwrite=False, hash_value=None,
                      hash_type="sha256", logger=None):
    """Download file, with logic (from tensor2tensor) for Google Drive. Returns
    the path to the downloaded file.

    Args:
        url: the url of the file from URL header. (None)
        root: download folder used to store the file in (.data)
        overwrite: overwrite existing files (False)
        hash_value (str, optional): hash for url (Default: ``None``).
        hash_type (str, optional): hash type, among "sha256" and "md5" (Default: ``"sha256"``).

    Examples:
        >>> url = 'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz'
        >>> kdmt.download.download_from_url(url)
        >>> url = 'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz'
        >>> kdmt.download.download_from_url(url)
        >>> '.data/validation.tar.gz'

    """
    if path is not None:
        path = os.path.abspath(path)
    root = os.path.abspath(root)

    def _check_hash(path):
        if hash_value:
            if logger:
                logger.info('Validating hash {} matches hash of {}'.format(hash_value, path))
            with open(path, "rb") as file_obj:
                if not validate_file_hash(file_obj, hash_value, hash_type):
                    raise RuntimeError("The hash of {} does not match. Delete the file manually and retry.".format(os.path.abspath(path)))

    def _process_response(r, root, filename):
        chunk_size = 16 * 1024
        total_size = int(r.headers.get('Content-length', 0))
        if filename is None:
            if 'content-disposition' not in r.headers:
                raise RuntimeError("Internal error: headers don't contain content-disposition.")
            d = r.headers['content-disposition']
            filename = re.findall("filename=\"(.+)\"", d)
            if filename is None:
                raise RuntimeError("Filename could not be autodetected")
            filename = filename[0]
        path = os.path.join(root, filename)
        if os.path.exists(path):
            if logger:
                logger.info('File %s already exists.' % path)
            if not overwrite:
                _check_hash(path)
                return path
            if logger:
                logger.info('Overwriting file %s.' % path)
        if logger:
            logger.info('Downloading file {} to {}.'.format(filename, path))
        with open(path, "wb") as file:
            with tqdm(total=total_size, unit='B',
                      unit_scale=1, desc=path.split('/')[-1]) as t:
                for chunk in r.iter_content(chunk_size):
                    if chunk:
                        file.write(chunk)
                        t.update(len(chunk))
        if logger:
            logger.info('File {} downloaded.'.format(path))

        _check_hash(path)
        return path

    if path is None:
        _, filename = os.path.split(url)
    else:
        root, filename = os.path.split(os.path.abspath(path))

    if not os.path.exists(root):
        try:
            os.makedirs(root)
        except OSError:
            print("Can't create the download directory {}.".format(root))
            raise

    if filename is not None:
        path = os.path.join(root, filename)
    # skip requests.get if path exists and not overwrite.
    if os.path.exists(path):
        if logger:
            logger.info('File %s already exists.' % path)
        if not overwrite:
            _check_hash(path)
            return path

    if 'drive.google.com' not in url:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, stream=True)
        return _process_response(response, root, filename)
    else:
        # google drive links get filename from google drive
        filename = None

    if logger:
        logger.info('Downloading from Google Drive; may take a few minutes')
    confirm_token = None
    session = requests.Session()
    response = session.get(url, stream=True)
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            confirm_token = v
    if confirm_token is None:
        if "Quota exceeded" in str(response.content):
            raise RuntimeError(
                "Google drive link {} is currently unavailable, because the quota was exceeded.".format(
                    url
                ))
        else:
            raise RuntimeError("Internal error: confirm_token was not found in Google drive link.")

    if confirm_token:
        url = url + "&confirm=" + confirm_token
        response = session.get(url, stream=True)

    return _process_response(response, root, filename)

