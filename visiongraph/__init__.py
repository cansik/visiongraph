"""
.. include:: ../README.md
.. include:: ../DOCUMENTATION.md
"""


def __getattr__(name):
    raise AttributeError(f"Visiongraph has no attribute '{name}'.\n\n"
                         f"Please note that with version '1.0.0', the 'vg' import has to be done like this:\n"
                         f"    from visiongraph import vg")
