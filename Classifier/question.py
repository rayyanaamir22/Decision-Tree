# Modules
import pandas as pd
from typing import Any

def isNumeric(val) -> bool:
    """
    Return True if <val> is a number (float or int).
    """
    return ((isinstance(val, int)) or (isinstance(val, float)))
class Question:
    """
    A Question is the object to be make the decision upon. It determines
    every break in the Tree.

    Attributes:
    ===========
    "colname": The name of the column which's value is being used in the decision, str
    "value": The value to be compared, any
    """
    colname: str
    value: Any

    def __init__(self, colname: str, value: Any) -> None:
        self.colname = colname
        self.value = value

    def __repr__(self) -> str:
        """
        Log the question for debugging.
        """
        condition = "==" if (not isNumeric(self.value)) else ">="
        return f"Is {self.colname} {condition} {self.value}"

    def match(self, example: pd.DataFrame) -> bool:
        """
        Compare value to one in another df example.
        """
        # This might not work if a Series object might not have column names
        # Should still work assuming it is inherited from the  DataFrame class
        val = example[self.colname] 
        if isNumeric(val):
            return val >= self.value
        return val == self.value