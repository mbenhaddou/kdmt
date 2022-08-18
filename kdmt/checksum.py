from hashlib import md5


def md5_hexdigest(file):
    """
    Calculate and return the MD5 checksum for a given file.
    ``file`` may either be a filename or an open stream.
    """
    if isinstance(file, str):
        with open(file, "rb") as infile:
            return _md5_hexdigest(infile)
    return _md5_hexdigest(file)


def _md5_hexdigest(fp):
    md5_digest = md5()
    while True:
        block = fp.read(1024 * 16)  # 16k blocks
        if not block:
            break
        md5_digest.update(block)
    return md5_digest.hexdigest()
