import pandas as pd 
import numpy as np 


def data_load() -> pd.DataFrame:
    titan_tf = pd.read_csv("../data/titanic_clean.csv")
    
    return titan_tf

    