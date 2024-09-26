from enum import Enum


class InputShapeOrder(Enum):
    """
    An enumeration to specify the order of input shape.

    Args:
        NCHW (int): The order with channels first and height/width second.
        NWHC (int): The order with channels first, width first and height second.
    """
    NCHW = 0
    NWHC = 1
