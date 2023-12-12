from data_preprocessing import *
from neural_network_model import *

'''
user_response = input("Which method do you want to use? (NN/CNN/random forest")

if user_response.lower()=='NN':
    method= 'NN'
elif user_response.lower()=='CNN':
    method='CNN'
'''


user_response = input("Dou you want to run Task 1 or Task 2? (1/2): ")

if user_response.lower() == '1':
    #data preprocessing
    trainLoader,testLoader, X_valid, Y_valid = data_preprocessing('NN', 'Task 1', 151)

    #tune hyperparameters
    print("TUNING")
    best_hyperparams, best_score = tune_nn_hyperparameters(X_valid, Y_valid, outputsize=7)

    #CREATE A BEST MODEL
    best_model = NeuralNetwork(hidden_size1 = best_hyperparams.hidden_size1, hidden_size2 = best_hyperparams.hidden_size2, output_size=7)
    torch.save(best_model, 'path/to/save/entire_model.pth')#change path
    
    #or use saved model
    #best_model=torch.load('best_model.pth')

    #train the best model
    best_model = train_nn_model(trainLoader, best_model, nbr_epoch=50)

    
    print("TESTING TUNED MODEL")
    test_nn_model(testLoader, best_model)

else: 
    #data preprocessing
    trainLoader,testLoader, X_valid, Y_valid = data_preprocessing('NN', 'Task 2', 151)

    #tune hyperparameters of the model
    print("TUNING")
    best_hyperparams, best_score = tune_nn_hyperparameters(X_valid, Y_valid, outputsize=38)

    #test model again with hyperparameters
    best_model = NeuralNetwork(hidden_size1 = best_hyperparams.hidden_size1, hidden_size2 = best_hyperparams.hidden_size2, output_size=38)
    torch.save(best_model, 'path/to/save/entire_model.pth')
    
    #or use saved model
    #best_model=torch.load('best_model.pth')
    
    #train best model
    best_model = train_nn_model(trainLoader, best_model)
    
    print("TESTING TUNED MODEL")
    test_nn_model(testLoader, best_model)