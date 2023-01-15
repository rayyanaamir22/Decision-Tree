from pytest import main
from helpers import uniqueValues, labelCounts, partition, gini, informationGain, findBestSplits
import pandas as pd

def test_uniqueValues() -> None:
    df = pd.DataFrame({
        "course" : ["CSC148", "MAT102", "MAT137"],
        "goal" : [90, 85, 85],
        "grade" : [None, None, None]
    })
    assert uniqueValues(df, "course") == {"CSC148", "MAT102", "MAT137"}
    assert uniqueValues(df, "goal") == {90, 85}

if __name__ == '__main__':
    main(["Classifier/Unit Tests/test_helpers.py"]) # Depends on open directory