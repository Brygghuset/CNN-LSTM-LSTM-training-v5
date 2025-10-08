"""
Imputeringsmetoder för saknade värden.

Definierar olika strategier för att hantera saknade värden i tidsseriedata.
"""

from enum import Enum


class ImputationMethod(Enum):
    """Enum för olika imputeringsmetoder."""
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    LINEAR_INTERPOLATION = "linear_interpolation"
    MEAN = "mean"
    MEDIAN = "median"
    ZERO = "zero"
    MASTER_POC_SMART_FORWARD_FILL = "master_poc_smart_forward_fill"  # Master POC standard 