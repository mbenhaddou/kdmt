import subprocess
import types
import importlib
import sys
from typing import Optional, Dict, Union
from distutils.version import LooseVersion
from importlib_metadata import distributions
from  logging import getLogger
from importlib import import_module


logger =getLogger()
INSTALLED_MODULES = None

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


def _try_import_and_get_module_version(
    modname: str,
) -> Optional[Union[LooseVersion, bool]]:
    """Returns False if module is not installed, None if version is not available"""
    try:
        if modname in sys.modules:
            mod = sys.modules[modname]
        else:

            mod = import_module(modname)
        try:
            ver = mod.__version__
        except AttributeError:
            # Version could not be obtained
            ver = None
    except ImportError:
        ver = False
    if ver:
        ver = LooseVersion(ver)
    return ver


# Based on packages_distributions() from importlib_metadata
def get_installed_modules() -> Dict[str, Optional[LooseVersion]]:
    """
    Get installed modules and their versions from pip metadata.
    """
    global INSTALLED_MODULES
    if not INSTALLED_MODULES:
        # Get all installed modules and their versions without
        # needing to import them.
        module_versions = {}
        # top_level.txt contains information about modules
        # in the package. It is not always present, in which case
        # the assumption is that the package name is the module name.
        # https://setuptools.pypa.io/en/latest/deprecated/python_eggs.html
        for dist in distributions():
            for pkg in (dist.read_text("top_level.txt") or "").split():
                try:
                    ver = LooseVersion(dist.metadata["Version"])
                except Exception:
                    ver = None
                module_versions[pkg] = ver
        INSTALLED_MODULES = module_versions
    return INSTALLED_MODULES


def _get_module_version(modname: str) -> Optional[Union[LooseVersion, bool]]:
    """Will cache the version in INSTALLED_MODULES

    Returns False if module is not installed."""
    installed_modules = get_installed_modules()
    if modname not in installed_modules:
        # Fallback. This should never happen unless module is not present
        installed_modules[modname] = _try_import_and_get_module_version(modname)
    return installed_modules[modname]


def get_module_version(modname: str) -> Optional[LooseVersion]:
    """Raises a ValueError if module is not installed"""
    version = _get_module_version(modname)
    if version is False:
        raise ValueError(f"Module '{modname}' is not installed.")
    return version


def is_module_installed(modname: str) -> bool:
    try:
        get_module_version(modname)
        return True
    except ValueError:
        return False



def check_dependencies(
    package: str,
    severity: str = "error",
    extra: Optional[str] = "all_extras",
    install_name: Optional[str] = None,
) -> bool:
    """Check if all soft dependencies are installed and raise appropriate error message
    when not.

    Parameters
    ----------
    package : str
        Package to check
    severity : str, optional
        Whether to raise an error ("error") or just a warning message ("warning"),
        by default "error"
    extra : Optional[str], optional
        The 'extras' that will install this package, by default "all_extras".
        If None, it means that the dependency is not available in optional
        requirements file and must be installed by the user on their own.
    install_name : Optional[str], optional
        The package name to install, by default None
        If none, the name in `package` argument is used

    Returns
    -------
    bool
        If error is set to "warning", returns True if package can be imported or False
        if it can not be imported

    Raises
    ------
    ModuleNotFoundError
        User friendly error with suggested action to install all required soft
        dependencies
    RuntimeError
        Is the severity argument is not one of the allowed values
    """
    install_name = install_name or package

    package_available = is_module_installed(package)

    if package_available:
        ver = get_module_version(package)
        print(
            "Soft dependency imported: {k}: {stat}".format(k=package, stat=str(ver))
        )
    else:
        msg = (
            f"\n'{package}' is a soft dependency and not included in the "
            f"pycaret installation. Please run: `pip install {install_name}` to install."
        )
        if extra is not None:
            msg += f"\nAlternately, you can install this by running `pip install pycaret[{extra}]`"

        if severity == "error":
            raise ModuleNotFoundError(msg)
        elif severity == "warning":
            print(f"Warning: {msg}")
            package_available = False
        else:
            raise RuntimeError(
                "Error in calling _check_soft_dependencies, severity "
                f'argument must be "error" or "warning", found "{severity}".'
            )

    return package_available


if __name__=="__main__":
    print(check_dependencies("tensorflow"))