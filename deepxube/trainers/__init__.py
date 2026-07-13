from importlib import import_module
from pkgutil import walk_packages

# Import all packages so that registers run
for _finder, name, _ipkg in walk_packages(__path__, prefix=__name__ + "."):
    import_module(name)
