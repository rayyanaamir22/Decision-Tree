# SOURCES:
# Writing a Decision Tree by Google Devs: https://www.youtube.com/watch?v=LDRbO9a6XPU&t=256s
# Google Devs Notebook Example: https://github.com/random-forests/tutorials/blob/master/decision_tree.ipynb
# Wikipedia: https://en.wikipedia.org/wiki/Decision_tree_learning

# Modules
import pandas as pd

# Other files
from question import Question

def uniqueValues(df: pd.DataFrame, colname: str) -> set:
    """
    Return the set of all distinct column names in <colname> of <df>.

    Example usage:
    >>> data = pd.DataFrame({
        "colour" : ["Green", "Yellow", "Red", "Red", "Yellow"],
        "diameter" : [3, 3, 1, 1, 3],
        "label" : ["Apple", "Apple", "Grape", "Grape", "Lemon"]
    })
    >>> uniqueValues(data)
    {'Apple', 'Grape', 'Lemon'}
    """
    return set(df[colname].unique())

def labelCounts(y: pd.Series):
    """
    Return a dict mapping each label in <colname> of <df> to its frequency.

    Example usage:
    >>> y = pd.Series(["Apple", "Apple", "Grape", "Grape", "Lemon"])
    >>> classCounts(data, "label")
    {"Yellow":2, "Red":2, "Green": 1} 
    """
    return dict(y.value_counts())

def partition(df: pd.DataFrame, 
              X: pd.DataFrame, 
              y: pd.Series, 
              question: Question
              ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    For every row in <df>, check if it matches the <question>.
    Return a df of the matching rows, followed by a df for those which
    did not match.

    Example Usage:
    >>> data = pd.DataFrame({
        "label" : ["Apple", "Apple", "Grape", "Grape", "Lemon"],
        "colour" : ["Green", "Yellow", "Red", "Red", "Yellow"],
        "diameter" : [3, 3, 1, 1, 3]
    })
    >>> t, f = partition(data, Question("colour", "Red")
    >>> t
    pd.DataFrame({
        "label" : ["Grape", "Grape"],
        "colour" : ["Red", "Red"],
        "diameter" : [1, 1]
    }) # The rows that match with the question
    >>> f
    pd.DataFrame({
        "label" : ["Apple", "Apple", "Lemon"],
        "colour" : ["Green", "Yellow", "Yellow"],
        "diameter" : [3, 3, 3]
    }) # The rows that do not match with the question


    """

    # Get indices
    trueIndices = question.match(df)
    falseIndices = ~trueIndices

    # True
    trueDf = X.loc[trueIndices]
    #trueLabels = y.loc[trueIndices]

    # False
    falseDf = X.loc[falseIndices]
    #falseLabels = y.loc[falseIndices]

    return trueDf, falseDf

def gini(y: pd.Series) -> float: # This might need to accept a <colname> param
    """
    Calculate the Gini Impurity for <y>.

    Gini Impurity is a metric that Decision Trees use to determine which
    questions are better to ask at a given break. Gini Impurity is defined as
    the probability of being incorrect if you randomly assign a label to an example
    in the same set.

    Example Usage:
    >>> noMixing = pd.Series(["Apple", "Apple"])
    >>> gini(noMixing)
    0.0 
    >>> someMixing = pd.Series(["Apple", "Orange"])
    >>> gini(someMixing)
    0.5
    """
    countsDict = labelCounts(y)
    impurity = 1
    totalEntries = float(y.shape[0]) # Number of rows
    for label in countsDict:
        probabilityOfLabel = countsDict[label] / totalEntries
        impurity -= probabilityOfLabel**2
    return impurity

def informationGain(
    left: pd.DataFrame, 
    right: pd.DataFrame, 
    y: pd.Series, 
    uncertainty: float
    ) -> float:
    """
    Return the Information Gain from choosing a given break.

    This is calculated as the <uncertainty> of the starting node, minus 
    the weighted impurity of the 2 child nodes <left> and <right>.

    Args:
    =====
    "left" (DataFrame): The left node of the break, consisting of all DataFrame rows on
            that side of the decision.
    "right" (DataFrame): The right node of the break, consisting of all DataFrame rows on
            that side of the decision.
    "uncertainty" (float): Uncertainty of the starting node as float.

    Return:
    =======
    The information gain of ... as float

    Example Usage:
    >>> idk bruh
    """

    # The weight is equal to the chance of a random sample landing 
    # in the <left> child node. This is also known as the Mutual Information score.
    weightOfLeft = float(left.shape[0]) / (left.shape[0] + right.shape[0])

    # Use the labels in y to calculate the Gini Impurity of the left and right node dfs
    return uncertainty - (weightOfLeft * gini(y.iloc[left.index])) - ((1-weightOfLeft) * gini(y.iloc[right.index]))

def findBestSplits(df: pd.DataFrame) -> tuple[float, Question]:
    """
    Find the best question to ask through a complete search of <df>. 
    "Best" refers to the split with the highest info gain.

    Args:
    =====
    "df" (DataFrame): The remaining rows of the full dataset which are to be split.

    Return:
    =======
    A tuple containing the highest information gain achievable, followed by
    the question that induces that gain.

    Example Usage:
    >>> data = pd.DataFrame({
        "colour" : ["Green", "Yellow", "Red", "Red", "Yellow"],
        "diameter" : [3, 3, 1, 1, 3],
        "label" : ["Apple", "Apple", "Grape", "Grape", "Lemon"]
    })
    >>> gain, q = findBestSplits(data)
    >>> gain
    0.37333333333333324
    >>> print(q)
    'Is colour == Red?'

    Note in the example above, if multiple questions yield the same 
    highest info gain, only the first question to yield that gain will 
    be returned.
    """

    # TRY USING RANDOM SUBSETS TO AVOID OVERFITTING.
    # Define a separate drop(rate) function.

    # Initialize these variables, they will (likely) change during the complete search
    bestGain = 0
    bestQuestion = None
    currentUncertainty = gini(df)
    
    # For each column
    Xcols = list(df.columns)
    Xcols.remove("label")
    for col in Xcols:
        # For each unique entry in that column
        for val in uniqueValues(df, col):
            currentQuestion = Question(col, val)
            
            # Attempt partition
            trueDf, falseDf = partition(df, currentQuestion)

            # If the split didn't divide anything, skip this iteration
            # because it will yield no information
            if not (trueDf.shape[0] or falseDf.shape[0]): # One split empty || other split full
                continue # Next iteration, skip the rest of this one

            currentGain = informationGain(trueDf, falseDf, currentUncertainty)
            if (currentGain > bestGain):
                bestGain, bestQuestion = currentGain, currentQuestion

    return bestGain, bestQuestion


if __name__ == '__main__':
    from doctest import testmod; testmod()