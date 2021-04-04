from contextlib import contextmanager
import sys
from progressbar import (ProgressBar, Percentage, Bar, ETA, FileTransferSpeed,
                         Timer, UnknownLength)




@contextmanager
def progress_bar(name, maxval):
    """Manages a progress bar for a download.

    Parameters
    ----------
    name : str
        Name of the downloaded file.
    maxval : int
        Total size of the download, in bytes.

    """
    if maxval is not UnknownLength:
        widgets = ['{}: '.format(name), Percentage(), ' ',
                   Bar(marker='=', left='[', right=']'), ' ', ETA(), ' ',
                   FileTransferSpeed()]
    else:
        widgets = ['{}: '.format(name), ' ', Timer(), ' ', FileTransferSpeed()]
    bar = ProgressBar(widgets=widgets, maxval=maxval, fd=sys.stdout).start()
    try:
        yield bar
    finally:
        bar.update(maxval)
        bar.finish()
