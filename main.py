from src.train import (data_load, transform_and_scaler, train_model)
from src.evaluate import evaluate_model

if __name__ == "__main__":
    
    # 1.Load
    titan_df = data_load()
    
    # 2.Transform and scale
    titan_df = transform_and_scaler(titan_df)
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = transform_and_scaler(titan_df)
    
    # 3. Train
    model_log, model_rfc, model_xgb = train_model(X_test_scaled, y_train)
    
    # 4. Evaluate
    evaluate = evaluate_model(model_log, model_rfc, model_xgb)
    