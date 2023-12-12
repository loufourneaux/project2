import torch.nn as nn
import torch.optim as optim
import torch
from skorch.callbacks import EarlyStopping
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV
import torch.nn.functional as F

class ConvNeuralNetwork(nn.Module):
    def __init__(self,input_channels = 3, kernel_size = (3,3,3), output_size = 7, hidden_size1 = 64, hidden_size2=128, hidden_size3 = 50, nbr_features = 99, stride = 1, padding = 1):
        super(ConvNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=input_channels, out_channels=hidden_size1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv3d(in_channels=hidden_size1, out_channels=hidden_size2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.fc1 = nn.Linear(in_features=nbr_features*hidden_size2, out_features=hidden_size3)
        self.fc2 = nn.Linear(in_features=hidden_size3, out_features=output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

modelCNN = ConvNeuralNetwork()

def get_cnn_model():
    return modelCNN

def restart_cnn_model(learning_rate = 0.0001):
    """

    :param learning_rate:
    :return:
    """

    modelCNN = ConvNeuralNetwork()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(modelCNN.parameters(), lr=learning_rate)
    return None

def train_cnn_model(trainLoader, model, learning_rate = 0.001, nbr_epoch = 100):
    """

    :param trainLoader:
    :param learning_rate:
    :param number_epoch:
    :return:
    """
    modelCNN = model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(modelCNN.parameters(), lr=learning_rate)
    modelCNN.train()

    all_loss = []
    all_accuracy = []
    all_epoch = []

    for epoch in range(nbr_epoch):
        running_loss = 0.0
        total_train = 0
        correct_train = 0

        for inputs, labels in trainLoader:
            optimizer.zero_grad()
            outputs = modelCNN(inputs)
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
        all_epoch.append(epoch+1)
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')
    return modelCNN, all_loss, all_accuracy, all_epoch

def test_cnn_model(testLoader, model):
    """

    :param testLoader:
    :param model:
    :return:
    """
    modelCNN = model
    modelCNN.eval()
    total_test = 0
    correct_test = 0

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in testLoader:
            labels = labels.long()
            outputs = modelCNN(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_accuracy = 100 * correct_test / total_test
    print(f'Accuracy on test set: {test_accuracy}%')

    return y_true,y_pred

def tune_cnn_hyperparameters(X_valid, y_valid, input_size = 99):
    #parameters to tune
    param_grid = {
        'module__hidden_size1': [4096,2048],
        'module__hidden_size2': [512],
        'batch_size': [153, 150, 152, 151],
        'optimizer__lr': [0.001]
    }

    modelCNN = ConvNeuralNetwork(input_channels = 3, kernel_size = (3,3,3), output_size = 7, hidden_size1 = 64, hidden_size2=128, hidden_size3 = 50,nbr_features = 99, stride = 1, padding = 1)
    modelCNN.eval()

    early_stopping = EarlyStopping(
        monitor='valid_loss',  # Change to 'valid_acc' for accuracy
        threshold=0.0001,       # Define your threshold
        threshold_mode='rel',  # 'rel' for relative, 'abs' for absolute
        patience=5            # Number of epochs to wait after condition is met
    )


    # Convert the PyTorch model to a skorch classifier to use in GridSearchCV
    classifier = NeuralNetClassifier(
        modelCNN,
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.Adam,
        max_epochs=50, # or choose an appropriate number of epochs
        callbacks=[early_stopping]
    )

    # Use GridSearchCV for hyperparameter tuning, cv for the number of folds in cross-validation, verbose for the explicit stage of tuning
    grid_search = GridSearchCV(classifier, param_grid, scoring='accuracy', cv=3, verbose=1)
    #get grid result
    grid_result= grid_search.fit(X_valid, y_valid)

    # Get the best hyperparameters
    best_hyperparams = grid_search.best_params_


    #get best score
    best_score=grid_search.best_score_

    return best_hyperparams, best_score