import re
import abc

from autogoal.utils import nice_repr
from autogoal.kb import Word, FeatureSet
from autogoal.grammar import BooleanValue
from autogoal.kb import AlgorithmBase


class _Regex(abc.ABC):
    def __init__(self, full: BooleanValue):
        self.full = full
        self._name = self.__class__.__name__[: -len("Regex")].lower()

    @abc.abstractmethod
    def _regex(self):
        pass

    def run(self, input: Word) -> FeatureSet:
        r_exp = self._regex()
        b = re.fullmatch(r_exp, input) if self.full else re.search(r_exp, input)
        return {f"is_{self._name}_regex": bool(b)}


@nice_repr
class UrlRegex(AlgorithmBase, _Regex):
    """
    Finds if a URL is contained inside a word using regular expressions.

    ##### Examples

    ```python
    >>> regex = UrlRegex(full=True)
    >>> regex.run("https://autogoal.gitlab.io/autogoal/contributing/#license")
    {'is_url_regex': True}

    >>> regex = UrlRegex(full=True)
    >>> regex.run("There is a URL at https://autogoal.gitlab.io/autogoal/contributing/#license, who would know?")
    {'is_url_regex': False}

    >>> regex = UrlRegex(full=False)
    >>> regex.run("There is a URL at https://autogoal.gitlab.io/autogoal/contributing/#license, who would know?")
    {'is_url_regex': True}

    ```
    """

    def _regex(self):
        return r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"


@nice_repr
class IPRegex(AlgorithmBase, _Regex):
    """
    Finds if an IP-address is contained inside a word using regular expressions.

    ##### Examples

    ```python
    >>> regex = IPRegex(full=True)
    >>> regex.run("192.168.18.1")
    {'is_ip_regex': True}

    >>> regex = IPRegex(full=True)
    >>> regex.run("There is an IP at 192.168.18.1, who would know?")
    {'is_ip_regex': False}

    >>> regex = IPRegex(full=False)
    >>> regex.run("There is an IP at 192.168.18.1, who would know?")
    {'is_ip_regex': True}

    ```
    """

    def _regex(self):
        return r"\b((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.)?){4}\b"


@nice_repr
class MACRegex(AlgorithmBase, _Regex):
    """
    Finds if a MAC-address is contained inside a word using regular expressions.

    ##### Examples

    ```python
    >>> regex = MACRegex(full=True)
    >>> regex.run("3D:F2:C9:A6:B3:4F")
    {'is_mac_regex': True}

    >>> regex = MACRegex(full=True)
    >>> regex.run("There is an IP at 3D-F2-C9-A6-B3-4F, who would know?")
    {'is_mac_regex': False}

    >>> regex = MACRegex(full=False)
    >>> regex.run("There is an IP at 3D:F2:C9:A6:B3:4F, who would know?")
    {'is_mac_regex': True}

    ```
    """

    def _regex(self):
        return r"([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})"


@nice_repr
class EmailRegex(AlgorithmBase, _Regex):
    """
    Finds if an email is contained inside a word using regular expressions.

    ##### Examples

    ```python
    >>> regex = EmailRegex(full=True)
    >>> regex.run("someone@example.com")
    {'is_email_regex': True}

    >>> regex = EmailRegex(full=True)
    >>> regex.run("There is an email at someone@example.com, who would know?")
    {'is_email_regex': False}

    >>> regex = EmailRegex(full=False)
    >>> regex.run("There is an email at someone@example.com, who would know?")
    {'is_email_regex': True}

    ```
    """

    def _regex(self):
        return r"([a-zA-Z0-9_\-\.]+)@([a-zA-Z0-9_\-\.]+)\.([a-zA-Z]{2,5})"


@nice_repr
class PhoneRegex(AlgorithmBase, _Regex):
    """
    Finds if a phone number is contained inside a word using regular expressions.

    ##### Examples

    ```python
    >>> regex = phoneRegex(full=True)
    >>> regex.run("+619123456789")
    {'is_phone_regex': True}

    >>> regex = phoneRegex(full=True)
    >>> regex.run("There is an phone at +619123456789, who would know?")
    {'is_phone_regex': False}

    >>> regex = phoneRegex(full=False)
    >>> regex.run("There is an phone at +619123456789, who would know?")
    {'is_phone_regex': True}

    ```
    """

    def _regex(self):
        return r"^((\+){1}91){1}[1-9]{1}[0-9]{9}$"
