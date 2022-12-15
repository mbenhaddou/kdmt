import errno
import json
import os
import pickle
import codecs
import gzip
import io
import tarfile
import six, sys, csv
import re, tempfile
import numpy as np
from kdmt.lib import install_and_import
try:
    import requests
except:
    install_and_import('requests')
import urllib.parse
import zipfile
try:
    import simplejson
except:
    install_and_import('simplejson')
import hashlib
import shutil




def list_directory(path):
    # type: (Text) -> List[Text]
    """Returns all files and folders excluding hidden files.

    If the path points to a file, returns the file. This is a recursive
    implementation returning files in any depth of the path."""

    if not isinstance(path, six.string_types):
        raise ValueError("Resourcename must be a string type")

    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        results = []
        for base, dirs, files in os.walk(path):
            # remove hidden files
            goodfiles = filter(lambda x: not x.startswith('.'), files)
            results.extend(os.path.join(base, f) for f in goodfiles)
        return results
    else:
        raise ValueError("Could not locate the resource '{}'."
                         "".format(os.path.abspath(path)))


def list_files(path):
    # type: (Text) -> List[Text]
    """Returns all files excluding hidden files.

    If the path points to a file, returns the file."""

    return [fn for fn in list_directory(path) if os.path.isfile(fn)]


def filename_from_url(url):
    """Parses a URL to determine a file name.

    Parameters
    ----------
    url : str
        URL to parse.

    """
    r = requests.get(url, stream=True)
    if 'Content-Disposition' in r.headers:
        filename = re.findall(r'filename=([^;]+)',
                              r.headers['Content-Disposition'])[0].strip('"\"')
    else:
        filename = os.path.basename(urllib.parse.urlparse(url).path)
    return filename


def read_file(filename, encoding="utf-8"):
    """Read text from a file."""
    with io.open(filename, encoding=encoding) as f:
        return f.read()


def read_json_file(filename, encoding="utf-8"):
    """Read json from a file."""
    content = read_file(filename, encoding=encoding)
    try:
        return simplejson.loads(content)
    except ValueError as e:
        raise ValueError("Failed to read json from '{}'. Error: "
                         "{}".format(os.path.abspath(filename), e))

def create_temporary_file(data, suffix="", mode="w+"):
    """Creates a tempfile.NamedTemporaryFile object for data.

    mode defines NamedTemporaryFile's  mode parameter in py3."""

    f = tempfile.NamedTemporaryFile(mode=mode, suffix=suffix,
                                    delete=False)
    f.write(data)

    f.close()
    return f.name

def jsonify_data(data):
    """Replaces JSON-non-serializable objects with JSON-serializable.

    Function replaces numpy arrays and numbers with python lists and numbers, tuples is replaces with lists. All other
    object types remain the same.

    Args:
        data: Object to make JSON-serializable.

    Returns:
        Modified input data.

    """
    if isinstance(data, (list, tuple)):
        result = [jsonify_data(item) for item in data]
    elif isinstance(data, dict):
        result = {}
        for key in data.keys():
            result[key] = jsonify_data(data[key])
    elif isinstance(data, np.ndarray):
        result = data.tolist()
    elif isinstance(data, np.integer):
        result = int(data)
    elif isinstance(data, np.floating):
        result = float(data)
    elif callable(getattr(data, "to_serializable_dict", None)):
        result = data.to_serializable_dict()
    else:
        result = data
    return result




def get_all_elems_from_json(search_json: dict, search_key: str) -> list:
    """Returns values by key in all nested dicts.

    Args:
        search_json: Dictionary in which one needs to find all values by specific key.
        search_key: Key for search.

    Returns:
        List of values stored in nested structures by ``search_key``.

    Examples:
        >>> get_all_elems_from_json({'a':{'b': [1,2,3]}, 'b':42}, 'b')
        [[1, 2, 3], 42]

    """
    result = []
    if isinstance(search_json, dict):
        for key in search_json:
            if key == search_key:
                result.append(search_json[key])
            else:
                result.extend(get_all_elems_from_json(search_json[key], search_key))
    elif isinstance(search_json, list):
        for item in search_json:
            result.extend(get_all_elems_from_json(item, search_key))

    return result


def create_dir(folder_name, force_perm=None):
    """Create the specified folder.

    If the parent folders do not exist, they are also created.
    If the folder already exists, nothing is done.

    Parameters
    ----------
    folder_name : str
        Name of the folder to create.
    force_perm : str
        Mode to use for folder creation.

    """
    if os.path.exists(folder_name):
        return
    intermediary_folders = folder_name.split(os.path.sep)

    # Remove invalid elements from intermediary_folders
    if intermediary_folders[-1] == "":
        intermediary_folders = intermediary_folders[:-1]
    if force_perm:
        force_perm_path = folder_name.split(os.path.sep)
        if force_perm_path[-1] == "":
            force_perm_path = force_perm_path[:-1]

    for i in range(1, len(intermediary_folders)):
        folder_to_create = os.path.sep.join(intermediary_folders[:i + 1])

        if os.path.exists(folder_to_create):
            continue
        os.mkdir(folder_to_create)
        if force_perm:
            os.chmod(folder_to_create, force_perm)

def json_to_string(obj, **kwargs):
    indent = kwargs.pop("indent", 2)
    ensure_ascii = kwargs.pop("ensure_ascii", False)
    return json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii, **kwargs)


def write_json_to_file(filename, obj, **kwargs):
    """Write an object as a json string to a file."""

    write_to_file(filename, json_to_string(obj, **kwargs))


def write_to_file(filename, text, encoding='utf-8'):
    """Write a text to a file."""

    with io.open(filename, 'w', encoding=encoding) as f:
        f.write(str(text))


def save_to_disk(path_to_disk, obj, overwrite=False):
    """ Pickle an object to disk """
    dirname = os.path.dirname(path_to_disk)
    if not os.path.exists(dirname):
        raise ValueError("Path " + dirname + " does not exist")

    if not overwrite and os.path.exists(path_to_disk):
        raise ValueError("File " + path_to_disk + "already exists")

    pickle.dump(obj, open(path_to_disk, 'wb'))


def load_from_disk(path_to_disk):
    """ Load a pickle from disk to memory """
    if not os.path.exists(path_to_disk):
        raise ValueError("File " + path_to_disk + " does not exist")

    return pickle.load(open(path_to_disk, 'rb'))

def binary_search_file(file, key, cache={}, cacheDepth=-1):
    """
    Return the line from the file with first word key.
    Searches through a sorted file using the binary search algorithm.
    :type file: file
    :param file: the file to be searched through.
    :type key: str
    :param key: the identifier we are searching for.
    """

    key = key + ' '
    keylen = len(key)
    start = 0
    currentDepth = 0

    if hasattr(file, 'name'):
        end = os.stat(file.name).st_size - 1
    else:
        file.seek(0, 2)
        end = file.tell() - 1
        file.seek(0)

    while start < end:
        lastState = start, end
        middle = (start + end) // 2

        if cache.get(middle):
            offset, line = cache[middle]

        else:
            line = ""
            while True:
                file.seek(max(0, middle - 1))
                if middle > 0:
                    file.discard_line()
                offset = file.tell()
                line = file.readline()
                if line != "":
                    break
                # at EOF; try to find start of the last line
                middle = (start + middle) // 2
                if middle == end - 1:
                    return None
            if currentDepth < cacheDepth:
                cache[middle] = (offset, line)

        if offset > end:
            assert end != middle - 1, "infinite loop"
            end = middle - 1
        elif line[:keylen] == key:
            return line
        elif line > key:
            assert end != middle - 1, "infinite loop"
            end = middle - 1
        elif line < key:
            start = offset + len(line) - 1

        currentDepth += 1
        thisState = start, end

        if lastState == thisState:
            # Detects the condition where we're searching past the end
            # of the file, which is otherwise difficult to detect
            return None

    return None


def read_yaml_string(string):
    import ruamel.yaml

    yaml_parser = ruamel.yaml.YAML(typ="safe")
    yaml_parser.version = "1.1"
    yaml_parser.unicode_supplementary = True

    return yaml_parser.load(string)


def read_yaml(fpath):
    from ruamel.yaml import YAML
    yaml = YAML(typ="safe")
    with open(fpath, encoding='utf8') as fin:
        return yaml.load(fin)

def _dump_yaml(obj, output):
    import ruamel.yaml

    yaml_writer = ruamel.yaml.YAML(pure=True, typ="safe")
    yaml_writer.unicode_supplementary = True
    yaml_writer.default_flow_style = False

    yaml_writer.dump(obj, output)


def dump_obj_as_yaml_to_file(filename, obj):
    """Writes data (python dict) to the filename in yaml repr."""
    with io.open(filename, 'w', encoding="utf-8") as output:
        _dump_yaml(obj, output)


def dump_obj_as_yaml_to_string(obj):
    """Writes data (python dict) to a yaml string."""
    str_io = io.StringIO()
    _dump_yaml(obj, str_io)
    return str_io.getvalue()


def dump_obj_as_str_to_file(filename, text):
    """Dump a text to a file."""

    with io.open(filename, 'w', encoding="utf-8") as f:
        # noinspection PyTypeChecker
        f.write(str(text))


def dump_obj_as_json_to_file(filename, obj):
    """Dump an object as a json string to a file."""

    dump_obj_as_str_to_file(filename, json.dumps(obj, indent=2))


def create_dir_for_file(file_path):
    """Creates any missing parent directories of this files path."""

    try:
        os.makedirs(os.path.dirname(file_path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def open_gziped(filename, mode='r', encoding=None):
    """Open a text file with encoding and optional gzip compression.

    Note that on legacy Python any encoding other than ``None`` or opening
    GZipped files will return an unpicklable file-like object.

    Parameters
    ----------
    filename : str
        The filename to read.
    mode : str, optional
        The mode with which to open the file. Defaults to `r`.
    encoding : str, optional
        The encoding to use (see the codecs documentation_ for supported
        values). Defaults to ``None``.

    .. _documentation:
    https://docs.python.org/3/library/codecs.html#standard-encodings

    """
    if filename.endswith('.gz') or filename.endswith('.tgz'):
        if six.PY2:
            zf = io.BufferedReader(gzip.open(filename, mode))
            if encoding:
                return codecs.getreader(encoding)(zf)
            else:
                return zf
        else:
            return io.BufferedReader(gzip.open(filename, mode,
                                               encoding=encoding))
    if six.PY2:
        if encoding:
            return codecs.open(filename, mode, encoding=encoding)
        else:
            return open(filename, mode)
    else:
        return open(filename, mode, encoding=encoding)


def tar_open(f):
    """Open either a filename or a file-like object as a TarFile.

    Parameters
    ----------
    f : str or file-like object
        The filename or file-like object from which to read.

    Returns
    -------
    TarFile
        A `TarFile` instance.

    """
    if isinstance(f, six.string_types):
        return tarfile.open(name=f)
    else:
        return tarfile.open(fileobj=f)

def find_in_data_path(filename, paths):
    """Searches for a file within on the paths.

    This function loops over all paths defined in Fuel's data path and
    returns the first path in which the file is found.

    Parameters
    ----------
    filename : str
        Name of the file to find.

    Returns
    -------
    file_path : str
        Path to the first file matching `filename` found in on the paths.

    Raises
    ------
    IOError
        If the file doesn't appear in one of the paths.

    """

    if isinstance(paths, str):
        paths=[paths]
    for path in paths:
        path = os.path.expanduser(os.path.expandvars(path))
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path):
            return file_path
    raise IOError("{} not found in the provided path".format(filename))

def ensure_directory_exists(directory):
    """Create directory (with parents) if does not exist, raise on failure.

    Parameters
    ----------
    directory : str
        The directory to create

    """
    if os.path.isdir(directory):
        return
    os.makedirs(directory)


def extract_archive(from_path, to_path=None, overwrite=False, logger=None):
    """Extract archive.

    Args:
        from_path: the path of the archive.
        to_path: the root path of the extracted files (directory of from_path)
        overwrite: overwrite existing files (False)

    Returns:
        List of paths to extracted files even if not overwritten.

    Examples:
    --------
        >>> url = 'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz'
        >>> from_path = './validation.tar.gz'
        >>> to_path = './'
        >>> kdmt.download.download_from_url(url, from_path)
        >>> kdmt.file.extract_archive(from_path, to_path)
        >>> ['.data/val.de', '.data/val.en']
        >>> kdmt.download.download_from_url(url, from_path)
        >>> kdmt.file.extract_archive(from_path, to_path)
        >>> ['.data/val.de', '.data/val.en']

    """

    if to_path is None:
        to_path = os.path.dirname(from_path)

    if from_path.endswith(('.tar.gz', '.tgz')):
        if logger:
            logger.info('Opening tar file {}.'.format(from_path))
        with tarfile.open(from_path, 'r') as tar:
            files = []
            for file_ in tar:
                file_path = os.path.join(to_path, file_.name)
                if '/._' in file_path:
                    continue
                if file_.isfile():
                    files.append(file_path)
                    if os.path.exists(file_path):
                        if logger:
                            logger.info('{} already extracted.'.format(file_path))
                        if not overwrite:
                            continue
                tar.extract(file_, to_path)
            if logger:
                logger.info('Finished extracting tar file {}.'.format(from_path))
            return files

    elif from_path.endswith('.zip'):
        assert zipfile.is_zipfile(from_path), from_path
        if logger:
            logger.info('Opening zip file {}.'.format(from_path))
        with zipfile.ZipFile(from_path, 'r') as zfile:
            files = []
            for file_ in zfile.namelist():
                file_path = os.path.join(to_path, file_)
                files.append(file_path)
                if os.path.exists(file_path):
                    if logger:
                        logger.info('{} already extracted.'.format(file_path))
                    if not overwrite:
                        continue
                zfile.extract(file_, to_path)
        files = [f for f in files if os.path.isfile(f)]
        if logger:
            logger.info('Finished extracting zip file {}.'.format(from_path))
        return files

    elif from_path.endswith('.gz'):
        if logger:
            logger.info('Opening gz file {}.'.format(from_path))
        default_block_size = 65536
        filename = from_path[:-3]
        files = [filename]
        with gzip.open(from_path, 'rb') as gzfile, \
                open(filename, 'wb') as d_file:
            while True:
                block = gzfile.read(default_block_size)
                if not block:
                    break
                else:
                    d_file.write(block)
            d_file.write(block)
        if logger:
            logger.info('Finished extracting gz file {}.'.format(from_path))
        return files

    else:
        raise NotImplementedError(
            "We currently only support tar.gz, .tgz, .gz and zip achives.")

def unicode_csv_reader(unicode_csv_data, **kwargs):
    r"""Since the standard csv library does not handle unicode in Python 2, we need a wrapper.
    Borrowed and slightly modified from the Python docs:
    https://docs.python.org/2/library/csv.html#csv-examples

    Args:
        unicode_csv_data: unicode csv data (see example below)

    Examples:
        >>> from kdmt.file import unicode_csv_reader
        >>> import io
        >>> with io.open(data_path, encoding="utf8") as f:
        >>>     reader = unicode_csv_reader(f)

    """

    # Fix field larger than field limit error
    maxInt = sys.maxsize
    while True:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt / 10)
    csv.field_size_limit(maxInt)

    for line in csv.reader(unicode_csv_data, **kwargs):
        yield line


def utf_8_encoder(unicode_csv_data):
    for line in unicode_csv_data:
        yield line.encode('utf-8')




def validate_file_hash(file_obj, hash_value, hash_type="sha256"):
    """Validate a given file object with its hash.

    Args:
        file_obj: File object to read from.
        hash_value (str): Hash for url.
        hash_type (str, optional): Hash type, among "sha256" and "md5" (Default: ``"sha256"``).
    Returns:
        bool: return True if its a valid file, else False.

    """

    if hash_type == "sha256":
        hash_func = hashlib.sha256()
    elif hash_type == "md5":
        hash_func = hashlib.md5()
    else:
        raise ValueError

    while True:
        # Read by chunk to avoid filling memory
        chunk = file_obj.read(1024 ** 2)
        if not chunk:
            break
        hash_func.update(chunk)
    return hash_func.hexdigest() == hash_value


def gz_extract(directory, delete=False):
    extension = ".gz"
    os.chdir(directory)
    for item in os.listdir(directory): # loop through items in dir
      if item.endswith(extension): # check for ".gz" extension
          gz_name = os.path.abspath(item) # get full path of files
          file_name = (os.path.basename(gz_name)).rsplit('.',1)[0] #get file name for file within
          with gzip.open(gz_name,"rb") as f_in, open(file_name,"wb") as f_out:
              shutil.copyfileobj(f_in, f_out)
          if delete:
            os.remove(gz_name) # delete zipped file




#gz_extract("/Users/mohamedmentis/Dropbox/Mac (2)/Documents/Mentis/Development/Python/Deep_kolibri/demos/current.tgz")