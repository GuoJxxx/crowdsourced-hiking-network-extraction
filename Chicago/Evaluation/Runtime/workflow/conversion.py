"""Select the Python 2 source translator available in the active interpreter."""
from importlib import import_module


def make_refactoring_tool():
    try:
        module=import_module('lib2to3.refactor')
        package='lib2to3.fixes'
    except ModuleNotFoundError as error:
        if error.name not in {'lib2to3','lib2to3.refactor'}:raise
        module=import_module('fissix.refactor')
        package='fissix.fixes'
    return module.RefactoringTool(module.get_fixers_from_package(package))
