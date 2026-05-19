from src.train import (data_load, transform_and_scaler, train_model)
from src.evaluate import evaluate_model

if __name__ == "__main__":
    
    # 1.Load
    titan_df = data_load()
    
    # 2.Transform and scale
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = transform_and_scaler(titan_df)

    # 3. Train
    model_log, model_rfc, model_xgb = train_model(X_train_scaled, y_train)
    
    # 4. Evaluate
    evaluate_model(model_log, model_rfc, model_xgb, X_test_scaled, y_test)