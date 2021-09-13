```py
    from sklearn.metrics import classification_report, f1_score, confusion_matrix, \
        accuracy_score, precision_score, recall_score
    
    predicted = model.predict(np.array([X_test[1]]))
    predicted_class = model.predict_classes(np.array([X_test[1]]))
    
    print('predicted:', predicted)
    print('predicted_class:', predicted_class)
    
    predicted_all = model.predict_classes(np.array(X_test))
    print('predicted_all:', predicted_all)
    
    # Have to convert one-hot encoding to actual classes: 0-9
    y_classes = ohe_to_classes(y_test)
    
    # string modulo operator
    #print("Accuracy: %.4f%%" % (accuracy_score(predicted_all, y_classes)*100))
    # using str.format()
    print('Accuracy:', '{0:.4f}%'.format(accuracy_score(predicted_all, y_classes)*100))
    #print('Precision', precision_score(predicted_all, y_classes))
    #print('Recall', recall_score(predicted_all, y_classes))
    #print('F1 Score', f1_score(predicted_all, y_classes))
    
    print("\nConufusion Matrix:\n", confusion_matrix(predicted_all, y_classes), "\n")
    print("Classification Report:\n", classification_report(predicted_all, y_classes))
```
