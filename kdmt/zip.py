from zipfile import ZipFile, ZIP_DEFLATED
import os
from typing import Union, Optional
from pathlib import Path
import tarfile
import gzip

def zipdir(path, zip_file_name, exclude_list=[]):
    # ziph is zipfile handle
    ziph = ZipFile(zip_file_name, 'w', ZIP_DEFLATED)
    for root, dirs, files in os.walk(path):
        for file in files:
            if file not in exclude_list:
                f=os.path.join(root, file)
                ziph.write(f, os.path.relpath(f, path))

    ziph.close()



def untar(file_path: Union[Path, str], extract_folder: Optional[Union[Path, str]] = None) -> None:
    """Simple tar archive extractor.

    Args:
        file_path: Path to the tar file to be extracted.
        extract_folder: Folder to which the files will be extracted.

    """
    file_path = Path(file_path)
    if extract_folder is None:
        extract_folder = file_path.parent
    extract_folder = Path(extract_folder)
    tar = tarfile.open(file_path)
    tar.extractall(extract_folder)
    tar.close()


def ungzip(file_path: Union[Path, str], extract_path: Optional[Union[Path, str]] = None) -> None:
    """Simple .gz archive extractor.

    Args:
        file_path: Path to the gzip file to be extracted.
        extract_path: Path where the file will be extracted.

    """
    chunk_size = 16 * 1024
    file_path = Path(file_path)
    if extract_path is None:
        extract_path = file_path.with_suffix('')
    extract_path = Path(extract_path)

    with gzip.open(file_path, 'rb') as fin, extract_path.open('wb') as fout:
        while True:
            block = fin.read(chunk_size)
            if not block:
                break
            fout.write(block)