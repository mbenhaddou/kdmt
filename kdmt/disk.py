
import os


def disk_usage(path):
    """Return free usage about the given path, in bytes.

    Parameters
    ----------
    path : str
        Folder for which to return disk usage

    Returns
    -------
    output : tuple
        Tuple containing total space in the folder and currently
        used space in the folder

    """
    st = os.statvfs(path)
    total = st.f_blocks * st.f_frsize
    used = (st.f_blocks - st.f_bfree) * st.f_frsize
    return total, used

def check_enough_space(dataset_local_dir, remote_fname, local_fname,
                       max_disk_usage=0.9):
    """Check if the given local folder has enough space.

    Check if the given local folder has enough space to store
    the specified remote file.

    Parameters
    ----------
    remote_fname : str
        Path to the remote file
    remote_fname : str
        Path to the local folder
    max_disk_usage : float
        Fraction indicating how much of the total space in the
        local folder can be used before the local cache must stop
        adding to it.

    Returns
    -------
    output : boolean
        True if there is enough space to store the remote file.

    """
    storage_need = os.path.getsize(remote_fname)
    storage_total, storage_used = disk_usage(dataset_local_dir)

    # Instead of only looking if there's enough space, we ensure we do not
    # go over max disk usage level to avoid filling the disk/partition
    return ((storage_used + storage_need) <
            (storage_total * max_disk_usage))
