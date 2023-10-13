import re
import pkg_resources
import importlib.util


def find_packages(regex):
    """
    Finds every installed package in current python enviroment based on its name matching the `regex` argument.

    Example usage:

    >>> matching_packages = find_packages(r'^numpy')
    >>> print(matching_packages)
    [numpy 1.24.2 (/usr/local/lib/python3.9/site-packages)]
    """
    installed_packages = [d for d in pkg_resources.working_set]
    pattern = re.compile(regex)
    matching_packages = [
        package for package in installed_packages if pattern.search(str(package))
    ]
    return matching_packages


def is_package_installed(package_name):
    # This function returns True if the package is installed, False otherwise
    return importlib.util.find_spec(package_name) is not None
