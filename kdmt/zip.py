from zipfile import ZipFile, ZIP_DEFLATED
import os
def zipdir(path, zip_file_name, exclude_list=[]):
    # ziph is zipfile handle
    ziph = ZipFile(zip_file_name, 'w', ZIP_DEFLATED)
    for root, dirs, files in os.walk(path):
        for file in files:
            if file not in exclude_list:
                f=os.path.join(root, file)
                ziph.write(f, os.path.relpath(f, path))

    ziph.close()


