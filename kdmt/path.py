#from etils.epath import *
import os, stat
from pathlib import Path

def is_writable(path):
    # Ensure that it exists.
    if not os.path.exists(path):
        return False

    # If we're on a posix system, check its permissions.
    if hasattr(os, "getuid"):
        statdata = os.stat(path)
        perm = stat.S_IMODE(statdata.st_mode)
        # is it world-writable?
        if perm & 0o002:
            return True
        # do we own it?
        elif statdata.st_uid == os.getuid() and (perm & 0o200):
            return True
        # are we in a group that can write to it?
        elif (statdata.st_gid in [os.getgid()] + os.getgroups()) and (perm & 0o020):
            return True
        # otherwise, we can't write to it.
        else:
            return False

    # Otherwise, we'll assume it's writable.
    # [xx] should we do other checks on other platforms?
    return True



def expand_path(path):
    """Convert relative paths to absolute with resolving user directory."""
    return Path(path).expanduser().resolve()