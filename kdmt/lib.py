import subprocess
import types
import importlib
import sys

class LazyLoader(types.ModuleType):
    """
    Lazily import a module, mainly to avoid pulling in large dependencies. For examples of modules that are large and not always needed, and this allows them to
    only be loaded when they are used.

    Example:
    --------

    >>> requests = LazyLoader('requests', globals(), 'requests')
    """

    def __init__(self, local_name, parent_module_globals, name):
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals

        super(LazyLoader, self).__init__(name)

    def _load(self):
        """ Load the module and insert it into the parent's globals. """

        # Import the target module and insert it into the parent's namespace
        module = importlib.import_module(self.__name__)
        self._parent_module_globals[self._local_name] = module

        # Update this object's dict so that if someone keeps a reference to the
        #   LazyLoader, lookups are efficient (__getattr__ is only called on lookups
        #   that fail).
        self.__dict__.update(module.__dict__)

        return module

    def __call__(self, *args, **kwargs):
        module = self._load()
        return module(*args, **kwargs)

    def __getattr__(self, item):
        module = self._load()
        return getattr(module, item)

    def __dir__(self):
        module = self._load()
        return dir(module)


def install_and_import(imported, package=None ):
    import importlib
    if package is None:
        package=imported
    try:
        importlib.import_module(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    finally:
        globals()[package] = importlib.import_module(imported)

