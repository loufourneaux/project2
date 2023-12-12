from Project2.data_preprocessing import *
from Project2.neural_network_model import *
from Project2.GRU_neural_network import *
from Project2.convolutional_neural_network import *
import time
from Project2.random_forest_classifier import *
from Project2.gradient_boosting_classifier import *


def time_elapsed(seconds):
    """

    :param seconds:
    :return:
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h} hours, {m} minutes, {s:.2f} seconds"


def task_1_nn():
    """

    :return:
    """
    start_time = time.time()

    # Data preprocessing
    print("Data Preprocessing...")
    trainLoader, testLoader, X_valid, Y_valid, output_size,input_size = data_preprocessing(model_name='NN', task_name='Task 1',
                                                                                batchsize=151, temp_size=0.95,
                                                                                test_size=0.95)
    print(f"Data preprocessing time: {time_elapsed(time.time() - start_time)}")

    # Tune hyperparameters
    modelNN = NeuralNetwork(output_size)
    start_time = time.time()
    print("Tuning hyperparameters...")
    best_hyperparams, best_score = tune_nn_hyperparameters(modelNN, X_valid, Y_valid, output_size, max_epochs=10)

    print(f"Tuning time: {time_elapsed(time.time() - start_time)}")

    # Create and save the best model
    start_time = time.time()
    print("Creating and saving the best model...")
    best_model = NeuralNetwork(output_size=output_size,input_size=input_size,
                               hidden_size1=best_hyperparams['module__hidden_size1'],
                               hidden_size2=best_hyperparams['module__hidden_size2'])
    torch.save(best_model, './entire_modelnn_task1.pth')  # change path
    print(f"Model creation and saving time: {time_elapsed(time.time() - start_time)}")

    # Train the best model
    start_time = time.time()
    print("Training the best model...")
    best_model,_,_,_ = train_nn_model(trainLoader, best_model, nbr_epoch=10)
    print(f"Training time: {time_elapsed(time.time() - start_time)}")

    # Testing the tuned model
    start_time = time.time()
    print("Testing tuned model...")
    test_nn_model(testLoader, best_model)
    print(f"Testing time: {time_elapsed(time.time() - start_time)}")


def task_1_gru():
    """

    :return:
    """
    start_time = time.time()

    # Data preprocessing
    print("Data Preprocessing...")
    trainLoader, testLoader, X_valid, Y_valid, output_size,input_size = data_preprocessing(model_name='GRU', task_name='Task 1',
                                                                                batchsize=151, temp_size=0.95,
                                                                                test_size=0.95)
    print(f"Data preprocessing time: {time_elapsed(time.time() - start_time)}")

    # Tune hyperparameters
    modelGRU = GRUNeuralNetwork(input_size=input_size, output_size=output_size)

    start_time = time.time()
    print("Tuning hyperparameters...")
    best_hyperparams, best_score = tune_gru_hyperparameters(modelGRU, X_valid, Y_valid, max_epochs=5)

    print(f"Tuning time: {time_elapsed(time.time() - start_time)}")

    # Create and save the best model
    start_time = time.time()
    print("Creating and saving the best model...")
    best_model = GRUNeuralNetwork(output_size=output_size,
                                  hidden_size1=best_hyperparams['module__hidden_size1'])
    torch.save(best_model, './entire_modelgru_task1.pth')  # change path
    print(f"Model creation and saving time: {time_elapsed(time.time() - start_time)}")

    # Train the best model
    start_time = time.time()
    print("Training the best model...")
    best_model,_,_,_ = train_gru_model(trainLoader, best_model, nbr_epoch=5)
    #best_model,_,_,_ = train_gru_model(trainLoader, modelGRU, nbr_epoch=10)
    print(f"Training time: {time_elapsed(time.time() - start_time)}")

    # Testing the tuned model
    start_time = time.time()
    print("Testing tuned model...")
    test_gru_model(testLoader, best_model)
    print(f"Testing time: {time_elapsed(time.time() - start_time)}")


def task_1_rf():
    """

    :return:
    """
    start_time = time.time()

    # Data preprocessing
    print("Data Preprocessing...")
    trainLoader, testLoader, X_valid, Y_valid, _, _ = data_preprocessing(model_name='RF', task_name='Task 1',
                                                             batchsize=151, temp_size=0.95,
                                                             test_size=0.95)

    print("Tuning hyperparameters...")
    best_hyperparams, best_score = tune_rf_hyperparameters(X_valid, Y_valid)
    print(f"Tuning time: {time_elapsed(time.time() - start_time)}")

    # Create and save the best model
    start_time = time.time()
    print("Creating and saving the best model...")
    best_model = RandomForestClassifier(**best_hyperparams)
    torch.save(best_model, './entire_modelrf_task1.pth')  # change path
    print(f"Model creation and saving time: {time_elapsed(time.time() - start_time)}")
    print(f"Data preprocessing time: {time_elapsed(time.time() - start_time)}")

    # Train and Test RandomForest Model
    start_time = time.time()
    print("Training and Testing RandomForest Model...")
    Y_pred_cv, y_pred_test, accuracy_train, accuracy_test = RF_cv(trainLoader, testLoader,best_hyperparams)

    print(f"Training and testing time: {time_elapsed(time.time() - start_time)}")
    print(f"Training Accuracy: {accuracy_train}")
    print(f"Testing Accuracy: {accuracy_test}")


def task_1_gbc():
    """

    :return:
    """
    start_time = time.time()

    # Data preprocessing
    print("Data Preprocessing...")
    trainLoader, testLoader, X_valid, Y_valid, _, _ = data_preprocessing(model_name='GBC', task_name='Task 1',
                                                                         batchsize=151, temp_size=0.95,
                                                                         test_size=0.95)

    print("Tuning hyperparameters...")
    best_hyperparams, best_score = tune_gbc_hyperparameters(X_valid, Y_valid)
    print(f"Tuning time: {time_elapsed(time.time() - start_time)}")

    # Create and save the best model
    start_time = time.time()
    print("Creating and saving the best model...")
    best_model = GradientBoostingClassifier(**best_hyperparams)
    torch.save(best_model, './entire_modelgbc_task1.pth')  # change path
    print(f"Model creation and saving time: {time_elapsed(time.time() - start_time)}")
    print(f"Data preprocessing time: {time_elapsed(time.time() - start_time)}")

    # Train and Test Gradient Boosting Descent Model
    start_time = time.time()
    print("Training and Testing Gradient Boosting Descent Model...")
    Y_pred_cv, y_pred_test, accuracy_train, accuracy_test = GBC_cv(trainLoader, testLoader,best_hyperparams)

    print(f"Training and testing time: {time_elapsed(time.time() - start_time)}")
    print(f"Training Accuracy: {accuracy_train}")
    print(f"Testing Accuracy: {accuracy_test}")


def task_1_cnn():
    """

    :return:
    """
    # data preprocessing
    trainLoader, testLoader, X_valid, Y_valid, output_size,input_size = data_preprocessing(model_name='CNN', task_name='Task 1'
                                                                                , batchsize=151, temp_size=0.95,
                                                                                test_size=0.95)
    modelCNN = ConvNeuralNetwork(input_channels=input_size, kernel_size=(3, 3, 3), output_size=output_size, hidden_size1=64,
                                 hidden_size2=128, hidden_size3=50, nbr_features=99, stride=1, padding=1)
    # tune hyperparameters
    print("TUNING")
    best_hyperparams, best_score = tune_cnn_hyperparameters(modelCNN, X_valid, Y_valid)

    # CREATE A BEST MODEL
    best_model = ConvNeuralNetwork(hidden_size1=best_hyperparams.hidden_size1,
                                   hidden_size2=best_hyperparams.hidden_size2, output_size=output_size,
                                   input_channels = input_size)
    torch.save(best_model, './entire_modelcnn_task1.pth')  # change path

    # or use saved model
    # best_model=torch.load('best_model.pth')

    # train the best model
    best_model,_,_,_ = train_cnn_model(trainLoader, best_model, nbr_epoch=50)

    print("TESTING TUNED MODEL")
    test_cnn_model(testLoader, best_model)


def task_2_nn():
    """

    :return:
    """
    start_time = time.time()

    # Data preprocessing
    print("Data Preprocessing...")
    trainLoader, testLoader, X_valid, Y_valid, output_size,input_size = data_preprocessing(model_name='NN', task_name='Task 2',
                                                                                           batchsize=151, temp_size=0.95,
                                                                                           test_size=0.95)
    print(f"Data preprocessing time: {time_elapsed(time.time() - start_time)}")
    print(output_size)
    # Tune hyperparameters
    modelNN = NeuralNetwork(output_size)
    start_time = time.time()
    print("Tuning hyperparameters...")
    best_hyperparams, best_score = tune_nn_hyperparameters(modelNN, X_valid, Y_valid,output_size, max_epochs=10)

    print(f"Tuning time: {time_elapsed(time.time() - start_time)}")

    # Create and save the best model
    start_time = time.time()
    print("Creating and saving the best model...")
    best_model = NeuralNetwork(output_size=output_size,input_size=input_size,
                               hidden_size1=best_hyperparams['module__hidden_size1'],
                               hidden_size2=best_hyperparams['module__hidden_size2'])
    torch.save(best_model, './entire_modelnn_task2.pth')  # change path
    print(f"Model creation and saving time: {time_elapsed(time.time() - start_time)}")

    # Train the best model
    start_time = time.time()
    print("Training the best model...")
    best_model,_,_,_ = train_nn_model(trainLoader, best_model, nbr_epoch=10)
    print(f"Training time: {time_elapsed(time.time() - start_time)}")

    # Testing the tuned model
    start_time = time.time()
    print("Testing tuned model...")
    test_nn_model(testLoader, best_model)
    print(f"Testing time: {time_elapsed(time.time() - start_time)}")


def task_2_gru():
    """

    :return:
    """
    start_time = time.time()

    # Data preprocessing
    print("Data Preprocessing...")
    trainLoader, testLoader, X_valid, Y_valid, output_size,input_size = data_preprocessing(model_name='GRU', task_name='Task 2',
                                                                                           batchsize=151, temp_size=0.95,
                                                                                           test_size=0.95)
    print(f"Data preprocessing time: {time_elapsed(time.time() - start_time)}")

    # Tune hyperparameters
    modelGRU = GRUNeuralNetwork(input_size=input_size, output_size=output_size)

    start_time = time.time()
    print("Tuning hyperparameters...")
    best_hyperparams, best_score = tune_gru_hyperparameters(modelGRU, X_valid, Y_valid, max_epochs=5)

    print(f"Tuning time: {time_elapsed(time.time() - start_time)}")

    # Create and save the best model
    start_time = time.time()
    print("Creating and saving the best model...")
    best_model = GRUNeuralNetwork(output_size=output_size,
                                  hidden_size1=best_hyperparams['module__hidden_size1'])
    torch.save(best_model, './entire_modelgru_task2.pth')  # change path
    print(f"Model creation and saving time: {time_elapsed(time.time() - start_time)}")

    # Train the best model
    start_time = time.time()
    print("Training the best model...")
    best_model,_,_,_ = train_gru_model(trainLoader, best_model, nbr_epoch=5)
    #best_model,_,_,_ = train_gru_model(trainLoader, modelGRU, nbr_epoch=10)
    print(f"Training time: {time_elapsed(time.time() - start_time)}")

    # Testing the tuned model
    start_time = time.time()
    print("Testing tuned model...")
    test_gru_model(testLoader, best_model)
    print(f"Testing time: {time_elapsed(time.time() - start_time)}")


def task_2_rf():
    """

    :return:
    """
    start_time = time.time()

    # Data preprocessing
    print("Data Preprocessing...")
    trainLoader, testLoader, X_valid, Y_valid, _, _ = data_preprocessing(model_name='RF', task_name='Task 2',
                                                                         batchsize=151, temp_size=0.95,
                                                                         test_size=0.95)

    print("Tuning hyperparameters...")
    best_hyperparams, best_score = tune_rf_hyperparameters(X_valid, Y_valid)
    print(f"Tuning time: {time_elapsed(time.time() - start_time)}")

    # Create and save the best model
    start_time = time.time()
    print("Creating and saving the best model...")
    best_model = RandomForestClassifier(**best_hyperparams)
    torch.save(best_model, './entire_modelrf_task2.pth')  # change path
    print(f"Model creation and saving time: {time_elapsed(time.time() - start_time)}")
    print(f"Data preprocessing time: {time_elapsed(time.time() - start_time)}")

    # Train and Test RandomForest Model
    start_time = time.time()
    print("Training and Testing RandomForest Model...")
    Y_pred_cv, y_pred_test, accuracy_train, accuracy_test = RF_cv(trainLoader, testLoader,best_hyperparams)

    print(f"Training and testing time: {time_elapsed(time.time() - start_time)}")
    print(f"Training Accuracy: {accuracy_train}")
    print(f"Testing Accuracy: {accuracy_test}")


def task_2_gbc():
    """

    :return:
    """
    start_time = time.time()

    # Data preprocessing
    print("Data Preprocessing...")
    trainLoader, testLoader, X_valid, Y_valid, _, _ = data_preprocessing(model_name='GBC', task_name='Task 2',
                                                                         batchsize=151, temp_size=0.95,
                                                                         test_size=0.95)

    print("Tuning hyperparameters...")
    best_hyperparams, best_score = tune_gbc_hyperparameters(X_valid, Y_valid)
    print(f"Tuning time: {time_elapsed(time.time() - start_time)}")

    # Create and save the best model
    start_time = time.time()
    print("Creating and saving the best model...")
    best_model = GradientBoostingClassifier(**best_hyperparams)
    torch.save(best_model, './entire_modelgbc_task2.pth')  # change path
    print(f"Model creation and saving time: {time_elapsed(time.time() - start_time)}")
    print(f"Data preprocessing time: {time_elapsed(time.time() - start_time)}")

    # Train and Test Gradient Boosting Descent Model
    start_time = time.time()
    print("Training and Testing Gradient Boosting Descent Model...")
    Y_pred_cv, y_pred_test, accuracy_train, accuracy_test = GBC_cv(trainLoader, testLoader,best_hyperparams)

    print(f"Training and testing time: {time_elapsed(time.time() - start_time)}")
    print(f"Training Accuracy: {accuracy_train}")
    print(f"Testing Accuracy: {accuracy_test}")


def task_2_cnn():
    """

    :return:
    """
    # data preprocessing
    trainLoader, testLoader, X_valid, Y_valid, output_size,input_size = data_preprocessing(model_name='CNN', task_name='Task 2'
                                                                                           , batchsize=151, temp_size=0.95,
                                                                                           test_size=0.95)
    modelCNN = ConvNeuralNetwork(input_channels=input_size, kernel_size=(3, 3, 3), output_size=output_size, hidden_size1=64,
                                 hidden_size2=128, hidden_size3=50, nbr_features=99, stride=1, padding=1)
    # tune hyperparameters
    print("TUNING")
    best_hyperparams, best_score = tune_cnn_hyperparameters(modelCNN, X_valid, Y_valid,output_size)

    # CREATE A BEST MODEL
    best_model = ConvNeuralNetwork(hidden_size1=best_hyperparams.hidden_size1,
                                   hidden_size2=best_hyperparams.hidden_size2, output_size=output_size,
                                   input_channels = input_size)
    torch.save(best_model, './entire_modelcnn_task2.pth')  # change path

    # or use saved model
    # best_model=torch.load('best_model.pth')

    # train the best model
    best_model,_,_,_ = train_cnn_model(trainLoader, best_model, nbr_epoch=50)

    print("TESTING TUNED MODEL")
    test_cnn_model(testLoader, best_model)


def run_task(task, model):
    """

    :param task:
    :param model:
    :return:
    """
    if task == 1:
        if model == 'NN':
            task_1_nn()
        elif model == 'GRU':
            task_1_gru()
        elif model == 'RF':
            task_1_rf()
        elif model == 'GBC':
            task_1_gbc()
        elif model == 'CNN':
            task_1_cnn()
    elif task == 2:
        if model == 'NN':
            task_2_nn()
        elif model == 'GRU':
            task_2_gru()
        elif model == 'RF':
            task_2_rf()
        elif model == 'GBC':
            task_2_gbc()
        elif model == 'CNN':
            task_2_cnn()


if __name__ == "__main__":
    while True:
        try:
            # Ask the user for the task number
            task_input = input("Enter the task number (1 or 2), 'exit' or ctrl+C to quit: ")

            # Check if the user wants to exit the program
            if task_input.lower() == 'exit':
                print("Exiting the program.")
                break

            # Validate the task number
            if task_input not in ['1', '2']:
                print("Invalid task number. Please enter a valid task number.")
                continue

            # Convert task input to an integer for further processing
            task = int(task_input)

            while True:
                # Ask the user for the model type
                model_input = input("Enter the model (NN, GRU, RF, GBC, CNN), 'exit' to come back to the task choice: ")

                # Check if the user wants to exit the program
                if model_input.lower() == 'exit':
                    print("Choose again the task.")
                    break  # Break out of the while loop to exit the program

                # Check for valid model
                if model_input not in ['NN', 'GRU', 'RF', 'GBC', 'CNN']:
                    print("Invalid model type. Please enter a valid model.")
                    continue  # Continue to the next iteration of the loop
                else:
                    # Run the task with the given model
                    run_task(task, model_input)
                    break

        except ValueError:
            print("An error occurred.")
        except Exception as e:
            # Handle other potential exceptions
            print(f"An error occurred: {e}")
        except KeyboardInterrupt:
            print("\nUser interrupted the program. Exiting...")
            break
