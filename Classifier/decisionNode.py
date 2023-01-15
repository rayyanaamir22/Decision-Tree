# Modules
import pandas as pd

# Other files
from question import Question


class DecisionNode:
    """
    A Decision Node poses a question. 

    This contains a reference to the question and the two child nodes.
    """

    def __init__(self, question: Question, trueDf: pd.DataFrame, falseDf: pd.DataFrame) -> None:
        self.question = question
        self.trueDf = trueDf
        self.falseDf = falseDf