import torch.nn as nn
import torch.optim as optim
import torch
from skorch.callbacks import EarlyStopping
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV


class GRUNeuralNetwork(nn.Module):
    def __init__(self, output_size, input_size=102, hidden_size1=64, activation='relu'):
        super(GRUNeuralNetwork, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size1)  # , batch_first=True)
        self.activation = self.get_activation(activation)
        self.layer2 = nn.Linear(hidden_size1, output_size)

    def forward(self, x):
        # x est de la forme (batch_size, sequence_length, input_size)
        output, _ = self.gru(x)
        # Prenez seulement la sortie de la dernière étape de la séquence
        # output = output[:, -1, :]
        output = self.activation(output)
        output = self.layer2(output)
        return output
    
    def get_activation(self, activation):
        if activation=='relu': 
            return nn.ReLU()
        elif activation=='sigmoid':
            return nn.Sigmoid()
        elif activation =='tanh':
            return nn.Tanh()
        else: raise ValueError(f"Unsupported activation function")


def restart_gru_model(learning_rate=0.0001):
    """

    :param learning_rate:
    :return:
    """

    modelGRU = GRUNeuralNetwork()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(modelGRU.parameters(), lr=learning_rate)


def train_gru_model(trainLoader, model, learning_rate=0.001, nbr_epoch=100):
    """

    :param trainLoader:
    :param learning_rate:
    :param number_epoch:
    :return:
    """
    modelGRU = model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(modelGRU.parameters(), lr=learning_rate)
    modelGRU.train()
    all_loss = []
    all_accuracy = []
    all_epoch = []
    for epoch in range(nbr_epoch):
        running_loss = 0.0
        total_train = 0
        correct_train = 0

        for inputs, labels in trainLoader:
            optimizer.zero_grad()
            outputs = modelGRU(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(trainLoader)
        epoch_accuracy = 100 * correct_train / total_train
        all_loss.append(epoch_loss)
        all_accuracy.append(epoch_accuracy)
        all_epoch.append(epoch + 1)

        print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')
    return modelGRU, all_loss, all_accuracy, all_epoch


def test_gru_model(testLoader, modelGRU):
    """

    :param testLoader:
    :param model:
    :return:
    """
    modelGRU.eval()
    total_test = 0
    correct_test = 0

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in testLoader:
            labels = labels.long()
            outputs = modelGRU(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_accuracy = 100 * correct_test / total_test
    print(f'Accuracy on test set: {test_accuracy}%')

    return y_true, y_pred


def tune_gru_hyperparameters(model, X_valid, y_valid, output_size, hidden_size1 = 64, threshold=0.0001, patience=5, max_epochs=50, cv=3, activation='relu',
                             verbose=1):
    """

    :param model:
    :param X_valid:
    :param y_valid:
    :param hidden_size1:
    :param threshold:
    :param patience:
    :param max_epochs:
    :param cv:
    :param verbose:
    :return:
    """
    # parameters to tune
    param_grid = {
        'module__hidden_size1': [4096, 2048],
        'optimizer__lr': [0.001],
        'module__activation':['relu','sigmoid','tanh']
    }
    modelGRU = model
    modelGRU.eval()

    early_stopping = EarlyStopping(
        monitor='valid_acc',  # Change to 'valid_acc' for accuracy
        threshold=threshold,  # Define your threshold
        threshold_mode='abs',  # 'rel' for relative, 'abs' for absolute
        patience=patience,  # Number of epochs to wait after condition is met
        lower_is_better=False# Number of epochs to wait after condition is met
    )

    # Convert the PyTorch model to a skorch classifier to use in GridSearchCV
    classifier = NeuralNetClassifier(
        module=GRUNeuralNetwork,
        module__hidden_size1=hidden_size1,  # Example values
        module__activation=activation,
        module__output_size=output_size,
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.Adam,
        max_epochs=max_epochs,  # or choose an appropriate number of epochs
        callbacks=[early_stopping]
    )

    # Use GridSearchCV for hyperparameter tuning, cv for the number of folds in cross-validation, verbose for the explicit stage of tuning
    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, scoring='accuracy', cv=cv, verbose=2, n_jobs=-1)
    # get grid result
    grid_result = grid_search.fit(X_valid, y_valid)

    # Get the best hyperparameters
    best_hyperparams = grid_search.best_params_

    # get best score
    best_score = grid_search.best_score_

    return best_hyperparams, best_score
