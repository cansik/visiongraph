import functools
import inspect
import traceback
import warnings


def deprecated(message: str = ""):
    """
    A decorator to mark functions as deprecated.

    :param message: An optional message indicating the reason for deprecation.

    """

    def decorator_wrapper(func):
        @functools.wraps(func)
        def function_wrapper(*args, **kwargs):
            current_call_source = "|".join(traceback.format_stack(inspect.currentframe()))
            if current_call_source not in function_wrapper.last_call_source:
                warnings.warn("Function {} is now deprecated! {}".format(func.__name__, message),
                              category=DeprecationWarning, stacklevel=2)
                function_wrapper.last_call_source.add(current_call_source)

            return func(*args, **kwargs)

        function_wrapper.last_call_source = set()

        return function_wrapper

    return decorator_wrapper
