import argparse
from importlib.util import spec_from_file_location, module_from_spec


def get_model(filename: str):
    modname = filename.split("/")[-1].split(".")[0]
    print(modname, filename)
    spec = spec_from_file_location(modname, filename)
    if spec is None:
        raise ValueError(f"Import error:{filename}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.Model()


get_model(f"models/{model}.py")
