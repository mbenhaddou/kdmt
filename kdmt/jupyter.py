try:
    from IPython import get_ipython
except:
    pass

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'PyDevTerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False




def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True

def print_html(html):
    """
    Display() helper to print html code
    :param html: html code to be printed
    :return:
    """
    from IPython.core.display import display, HTML
    try:
        display(HTML(html))
        return True
    except NameError:
        return False

