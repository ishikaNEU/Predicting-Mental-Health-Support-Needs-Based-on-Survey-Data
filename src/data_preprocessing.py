
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    X = df[['Age', 'Gender', 'Work_Interfere', 'Benefits']]
    y = df['seek_help']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    numeric_features = ['Age']
    numeric_transformer = StandardScaler()
    
    categorical_features = ['Gender', 'Work_Interfere', 'Benefits']
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    
    X_train = pipeline.fit_transform(X_train)
    X_test = pipeline.transform(X_test)
    
    return X_train, X_test, y_train, y_test, pipeline

def save_pipeline(pipeline, filename):
    with open(filename, 'wb') as f:
        pickle.dump(pipeline, f)
