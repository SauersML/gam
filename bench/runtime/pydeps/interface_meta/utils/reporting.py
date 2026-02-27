import logging

from .errors import InterfaceConformanceError


def report_violation(message, raise_on_violation):
    """
    Report a violation in conformance to the user.

    Args:
        message (str): The message to pass on to the user.
        raise_on_violation (bool): Whether any non-conformance should cause an
            exception to be raised. (default: False)
    """
    if raise_on_violation:
        raise InterfaceConformanceError(message)
    else:
        logging.warning(message)
