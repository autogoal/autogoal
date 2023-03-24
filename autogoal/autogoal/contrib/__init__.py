from typing import List, Tuple
from autogoal.kb import AlgorithmBase
from autogoal.utils import find_packages
from os.path import abspath, dirname, join
from inspect import getsourcefile
import enum
import yaml

config_dir = dirname(abspath(getsourcefile(lambda: 0)))


def find_classes(include=None, exclude=None, modules=None, input=None, output=None):
    import inspect
    import re

    result = []

    if include:
        include = f".*({include}).*"
    else:
        include = r".*"

    if exclude:
        exclude = f".*({exclude}).*"

    if input:
        input = f".*({input}).*"

    if output:
        output = f".*({output}).*"

    if modules is None:
        modules = []

        for package in get_installed_contribs(exclude="remote"):
            modules.append(__import__(package.key))

    for module in modules:
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if not hasattr(cls, "run"):
                continue

            if cls.__name__.startswith("_"):
                continue

            if not re.match(include, repr(cls)):
                continue

            if exclude is not None and re.match(exclude, repr(cls)):
                continue

            if not issubclass(cls, AlgorithmBase):
                continue

            sig = inspect.signature(cls.run)

            if input and not re.match(input, str(sig.parameters["input"].annotation)):
                continue

            if output and not re.match(output, str(sig.return_annotation)):
                continue

            result.append(cls)

    return result


def find_remote_classes(
    sources: List[Tuple[str, int] or Tuple[str, int, str] or str] = None,
    include=None,
    exclude=None,
    input=None,
    output=None,
):
    try:
        autogoal_remote = __import__("autogoal-remote")
    except ImportError as e:
        print(
            "Module autogoal-remote not installed. To install it with all its dependencies run pip install autogoal-remote"
        )

    import itertools
    import re

    get_algorithms = autogoal_remote.get_algorithms
    store_connection = autogoal_remote.store_connection

    if include:
        include = f".*({include}).*"
    else:
        include = r".*"

    if exclude:
        exclude = f".*({exclude}).*"

    if input:
        input = f".*({input}).*"

    if output:
        output = f".*({output}).*"

    if len(sources) == 0:
        pass

    classes_by_contrib = {}
    for source in sources:
        s_type = type(source)
        ip = None
        port = None
        alias = source if s_type == str else None

        if s_type == tuple:
            if len(source) == 2:
                ip, port = source
            elif len(source) == 3:
                ip, port, alias = source
                store_connection(ip, port, alias)

        for contrib in itertools.groupby(
            get_algorithms(ip, port, alias), lambda x: x.contrib
        ):
            key, classes = contrib
            contrib_results = []

            for cls in classes:
                if not re.match(include, repr(cls)):
                    continue

                if exclude is not None and re.match(exclude, repr(cls)):
                    continue

                inp = cls.input_types()
                outp = cls.output_type()

                if input and not re.match(input, str(inp)):
                    continue

                if output and not re.match(output, str(outp)):
                    continue

                contrib_results.append(cls)
            classes_by_contrib[key] = contrib_results

    result = [cls for _, classes in classes_by_contrib.items() for cls in classes]
    return result


def get_registered_contribs():
    path = join(config_dir, "registered-contribs.yml")
    try:
        with open(path, "r") as fd:
            result = yaml.safe_load(fd)
    except IOError as e:
        result = []
    return result


def get_installed_contribs(exclude: str = None):
    """
    find all installed contrib modules.
    """

    packages_identifier = (
        "autogoal-" if exclude is None else rf"autogoal-(?!.*\b{exclude}\b).*"
    )
    modules = []
    for pkg in find_packages(packages_identifier):
        try:
            __import__(pkg.key)
            modules.append(pkg)
        except ImportError as e:
            print(
                f"Error importing {pkg.project_name} {pkg.version}. Use pip install {pkg.project_name} to ensure all dependencies are installed correctly."
            )
    return modules


class ContribStatus(enum.Enum):
    RequiresDependency = enum.auto()
    RequiresDownload = enum.auto()
    Ready = enum.auto()


def status():
    status = {}
    modules = []

    for pkg_name in get_registered_contribs():
        status[pkg_name] = ContribStatus.RequiresDependency

    for module in get_installed_contribs():
        status[module.project_name] = ContribStatus.Ready
        modules.append(module)

    modules.sort(key=lambda m: m.project_name)
    for module in modules:
        if hasattr(module, "status"):
            status[module.project_name] = module.status()

    return status


def download(contrib: str):
    modules = {}

    for package in get_installed_contribs():
        modules[package.key] = __import__(package.key)

    if contrib not in modules:
        raise ValueError(f"Contrib `{contrib}` cannot be imported.")

    contrib = modules[contrib]

    if not hasattr(contrib, "download"):
        return False

    return contrib.download()


__all__ = [
    "find_classes",
    "status",
    "download",
    "get_installed_contribs",
    "get_registered_contribs",
]
