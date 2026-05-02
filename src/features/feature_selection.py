import pandas as pd

def select_features(df):
    """
    Selects columns based on Spearman correlation and Mutual Information analysis.
    """
    # Column list based on the technical report
    winning_features = [
        'age', 'workclass', 'education_num', 'marital_status', 
        'occupation', 'relationship', 'race', 'sex', 
        'capital_gain', 'capital_loss', 'hours_per_week'
    ]
    
    return df[winning_features]