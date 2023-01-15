# Modules
import pandas as pd

# Other files
from helpers import classCounts

class Leaf:
    """
    A Leaf node on the Decision Tree. It serves the purpose of finally
    classifying the data once it has been reached.
    """
    def __init__(self, df: pd.DataFrame) -> None:
        self.predictions = classCounts(df, "label")

    def __repr__(self) -> str:
        return f"\nLabel counts: {self.predictions}\n"