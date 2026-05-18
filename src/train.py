import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def data_load() -> pd.DataFrame:
    titan_tf = pd.read_csv("../data/titanic_clean.csv")
    
    return titan_tf

    
def transform_and_scaler(titan_df: pd.DataFrame) -> tuple:
    X = titan_df.drop(['Survived', 'Name', 'Ticket', 'PassengerId'], axis=1)
    y = titan_df['Survived']
    
    X = pd.get_dummies(X, columns=['Sex', 'Embarked'], drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled