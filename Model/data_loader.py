import pandas as pd

def load_training_data():
    return pd.read_csv('../Data/test_input.csv')

def load_test_data():
    return pd.read_csv('../Data/test.csv')

# Not sure if these are right file names pls double check
