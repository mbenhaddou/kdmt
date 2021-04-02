import errno
import io
import json
import os
import pickle

import simplejson


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


def create_dir(dir_path):
    """Creates a directory and its super paths.

    Succeeds even if the path already exists."""

    try:
        os.makedirs(dir_path)
    except OSError as e:
        # be happy if someone already created the path
        if e.errno != errno.EEXIST:
            raise


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


def read_yaml_string(string):
    import ruamel.yaml

    yaml_parser = ruamel.yaml.YAML(typ="safe")
    yaml_parser.version = "1.1"
    yaml_parser.unicode_supplementary = True

    return yaml_parser.load(string)


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
