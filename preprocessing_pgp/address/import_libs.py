import importlib.util
import sys

# For illustrative purposes.
list_module = ['pandas', 'glob', 'multiprocessing', 'functools', 'numpy', 're', 'unidecode', 'time',
              'os', 'datetime', 'argparse', 'flashtext', 'itertools']

for name in list_module:
    try:
        if name in sys.modules:
            print(f"{name!r} already in sys.modules")
        elif (importlib.util.find_spec(name)) is not None:
            spec = importlib.util.find_spec(name)
            # If you choose to perform the actual import ...
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            print(f"{name!r} has been imported")
        else:
            print(f"can't find the {name!r} module")
    except Exception as e:
        print(str(e))