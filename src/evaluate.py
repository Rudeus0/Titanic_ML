from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score


def model_evaluate(model_log, model_rfc, X_test_scaled, y_test):
    
    y_pred_log = model_log.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred_log)
    print(f"Classification Accuracy:{acc:.4f}\n")
    
    cm = confusion_matrix(y_test, y_pred_log )
    print(f"\nConfusion Matrix :{cm}\n")
    
    cr = classification_report(y_test, y_pred_log)    
    print(f" \n Classificatioin Report:{ cr} \n")
    
    roc = roc_auc_score(y_test, model_log.predict_proba(X_test_scaled)[:, 1])
    print(f" \n Roc:{roc: .4f} ")
    

    
    y_pred_rfc = model_rfc.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred_rfc)
    print(f"\n Accuracy:{accuracy:.4f} \n")

    cm = confusion_matrix(y_test, y_pred_rfc)
    print(f"\n Confusion:{cm} \n ")

    cr = classification_report(y_test, y_pred_rfc)
    print(f" \n Classificatioin Report:{ cr} \n")

    roc = roc_auc_score(y_test, model_rfc.predict_proba(X_test_scaled)[:, 1])
    print(f" \n Roc:{roc: .4f} ")