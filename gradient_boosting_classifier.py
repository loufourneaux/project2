import torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


def GBC_cv2(trainLoader, testLoader, cv=5, n_estimators=100, random_state=100):
    # Combining the data from the trainLoader
    X_train, Y_train = [], []
    for data in trainLoader:
        inputs, labels = data
        X_train.append(inputs)
        Y_train.append(labels)

    X_train = torch.cat(X_train, 0).numpy()
    Y_train = torch.cat(Y_train, 0).numpy()

    # Training the Gradient Boosting Classifier
    gb_classifier = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state)
    Y_pred_cv = cross_val_predict(gb_classifier, X_train, Y_train, cv=cv)
    gb_classifier.fit(X_train, Y_train)

    # Combining the data from the testLoader
    X_test = []
    Y_test = []
    for data in testLoader:
        inputs, labels = data
        X_test.append(inputs)
        Y_test.append(labels)

    X_test = torch.cat(X_test, 0).numpy()

    # Predicting on the test set
    y_pred_test = gb_classifier.predict(X_test)
    accuracy_train = accuracy_score(Y_train, Y_pred_cv)
    accuracy_test = accuracy_score(Y_test, y_pred_test)

    return Y_pred_cv, y_pred_test, accuracy_train, accuracy_test

def GBC_cv(trainLoader, testLoader, best_params):
    # Combining the data from the trainLoader
    X_train, Y_train = [], []
    for data in trainLoader:
        inputs, labels = data
        X_train.append(inputs)
        Y_train.append(labels)

    X_train = torch.cat(X_train, 0).numpy()
    Y_train = torch.cat(Y_train, 0).numpy()

    # Training the RandomForest Classifier with best parameters
    gb_classifier = GradientBoostingClassifier(**best_params)
    Y_pred_cv = cross_val_predict(gb_classifier, X_train, Y_train, cv=5)
    gb_classifier.fit(X_train, Y_train)

    # Combining the data from the testLoader
    X_test = []
    Y_test = []
    for data in testLoader:
        inputs, labels = data
        X_test.append(inputs)
        Y_test.append(labels)

    X_test = torch.cat(X_test, 0).numpy()
    Y_test = torch.cat(Y_test, 0).numpy()

    # Predicting on the test set
    y_pred_test = gb_classifier.predict(X_test)
    accuracy_train = accuracy_score(Y_train, Y_pred_cv)
    accuracy_test = accuracy_score(Y_test, y_pred_test)

    return Y_pred_cv, y_pred_test, accuracy_train, accuracy_test

def tune_gbc_hyperparameters(X_validation, Y_validation, cv=5, verbose=2, n_jobs=-1):
    # Define the parameter grid to test
    param_grid = {
        'n_estimators': [100, 200, 300],
        'random_state': [100],
        'max_depth': [None, 10, 20, 30],
    }

    # Initialize RandomForest model
    gb_classifier = GradientBoostingClassifier()

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=gb_classifier, param_grid=param_grid, cv=cv, verbose=verbose, n_jobs=n_jobs)

    # Execute the grid search
    grid_search.fit(X_validation, Y_validation)

    return grid_search.best_params_, grid_search.best_score_