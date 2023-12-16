from data_preprocessing import *
from neural_network_model import *
from GRU_neural_network import *
from convolutional_neural_network import *
import time
from random_forest_classifier import *
from gradient_boosting_classifier import *
from visualization import *
import os
import joblib

def time_elapsed(seconds):
    """

    :param seconds:
    :return:
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h} hours, {m} minutes, {s:.2f} seconds"


def task_1_nn(save_type, model_file_path):
    """

    :return:
    """
    model_file = ''
    if save_type in ['t', 'tt']:
        model_file = model_file_path

    if save_type in ['t', 'tt'] and os.path.exists(model_file):
        print(f"Loading saved model from {model_file}")
        best_model = torch.load(model_file)

    start_time = time.time()

    # Data preprocessing
    print("Data Preprocessing...")
    trainLoader, testLoader, X_valid, Y_valid, output_size, input_size = data_preprocessing(model_name='NN',
                                                                                            task_name='Task 1',
                                                                                            batchsize=151,
                                                                                            temp_size=0.3,
                                                                                            test_size=0.5)
    print(f"Data preprocessing time: {time_elapsed(time.time() - start_time)}")

    if save_type == 'new':
        # Tune hyperparameters
        modelNN = NeuralNetwork(output_size)
        start_time = time.time()
        print("Tuning hyperparameters...")
        best_hyperparams, best_score = tune_nn_hyperparameters(modelNN, X_valid, Y_valid, output_size, max_epochs=10)

        print(f"Tuning time: {time_elapsed(time.time() - start_time)}")

        # Create and save the best model
        start_time = time.time()
        print("Creating and saving the best model...")
        best_model = NeuralNetwork(output_size=output_size, input_size=input_size,
                                   hidden_size1=best_hyperparams['module__hidden_size1'],
                                   hidden_size2=best_hyperparams['module__hidden_size2'])
        torch.save(best_model, './entire_modelnn_task1_tuned.pth')  # change path
        print(f"Model creation and saving time: {time_elapsed(time.time() - start_time)}")

    if save_type in ['new', 't']:
        # Train the best model
        start_time = time.time()
        print("Training the best model...")
        best_model, all_loss, all_accuracy, all_epoch = train_nn_model(trainLoader, best_model, nbr_epoch=3)
        print(f"Training time: {time_elapsed(time.time() - start_time)}")

        save_training_results_to_csv(all_loss,all_accuracy,all_epoch,"training_result_nn_1.csv")

        start_time = time.time()
        print("Saving tuned and trained model...")
        torch.save(best_model, './entire_modelnn_task1_tuned_trained.pth')  # change path
        print(f"Model creation and saving time: {time_elapsed(time.time() - start_time)}")

    # Testing the tuned model
    start_time = time.time()
    print("Testing tuned model...")
    y_true, y_pred = test_nn_model(testLoader, best_model)
    save_test_results_to_csv(y_true,y_pred, "test_result_nn_1.csv")
    print(f"Testing time: {time_elapsed(time.time() - start_time)}")


def task_1_gru(save_type, model_file_path):
    """

    :return:
    """
    model_file = ''
    if save_type in ['t', 'tt']:
        model_file = model_file_path

    if save_type in ['t', 'tt'] and os.path.exists(model_file):
        print(f"Loading saved model from {model_file}")
        best_model = torch.load(model_file)

    start_time = time.time()

    # Data preprocessing
    print("Data Preprocessing...")
    trainLoader, testLoader, X_valid, Y_valid, output_size, input_size = data_preprocessing(model_name='GRU',
                                                                                            task_name='Task 1',
                                                                                            batchsize=151,
                                                                                            temp_size=0.3,
                                                                                            test_size=0.5)
    print(f"Data preprocessing time: {time_elapsed(time.time() - start_time)}")

    # Tune hyperparameters
    if save_type == 'new':
        modelGRU = GRUNeuralNetwork(input_size=input_size, output_size=output_size)

        start_time = time.time()
        print("Tuning hyperparameters...")
        best_hyperparams, best_score = tune_gru_hyperparameters(modelGRU, X_valid, Y_valid, max_epochs=5)

        print(f"Tuning time: {time_elapsed(time.time() - start_time)}")

        # Create and save the best model
        start_time = time.time()
        print("Creating and saving the best model...")
        best_model = GRUNeuralNetwork(output_size=output_size,
                                      hidden_size1=best_hyperparams['module__hidden_size1'],
                                      activation=best_hyperparams['module_activation'])
        torch.save(best_model, './entire_modelgru_task1_tuned.pth')  # change path
        print(f"Model creation and saving time: {time_elapsed(time.time() - start_time)}")

    if save_type in ['new', 't']:

    # Train the best model
        start_time = time.time()
        print("Training the best model...")
        best_model, all_loss, all_accuracy, all_epoch = train_gru_model(trainLoader, best_model, nbr_epoch=50)
        print(f"Training time: {time_elapsed(time.time() - start_time)}")
        save_training_results_to_csv(all_loss,all_accuracy,all_epoch,"training_result_gru_1.csv")
        start_time = time.time()
        print("Saving tuned and trained model...")
        torch.save(best_model, './entire_modelgru_task1_tuned_trained.pth')  # change path
        print(f"Model creation and saving time: {time_elapsed(time.time() - start_time)}")

    # Testing the tuned model
    start_time = time.time()
    print("Testing tuned model...")
    y_true, y_pred = test_gru_model(testLoader, best_model)
    save_test_results_to_csv(y_true,y_pred, "training_result_gru_1.csv")
    print(f"Testing time: {time_elapsed(time.time() - start_time)}")


def task_1_rf(save_type, model_file_path):
    """

    :return:
    """

    model_file = ''
    if save_type in ['t', 'tt']:
        model_file = model_file_path

    if save_type in ['t', 'tt'] and os.path.exists(model_file):
        print(f"Loading saved model from {model_file}")
        best_model = joblib.load(model_file)
        best_hyperparams=best_model.get_params()
    else:
        start_time = time.time()

        # Data preprocessing
        print("Data Preprocessing...")
        trainLoader, testLoader, X_valid, Y_valid, _, _ = data_preprocessing(model_name='RF', task_name='Task 1',
                                                                             batchsize=152, temp_size=0.3,
                                                                             test_size=0.5)

        print("Tuning hyperparameters...")
        best_hyperparams, best_score = tune_rf_hyperparameters(X_valid, Y_valid)
        print(f"Tuning time: {time_elapsed(time.time() - start_time)}")

        # Create and save the best model
        start_time = time.time()
        print("Creating and saving the best model...")
        best_model = RandomForestClassifier(**best_hyperparams)
        joblib.dump(best_model,'./entire_modelrf_task1_tuned.joblib') # Saving tuned model

    # Train and Test RandomForest Model
    start_time = time.time()
    print("Training and Testing RandomForest Model...")
    y_train, y_test, Y_pred_cv, y_pred_test, accuracy_train, accuracy_test, train_tuned_model = RF_cv(trainLoader, testLoader, best_hyperparams)
    joblib.dump(train_tuned_model,'./entire_modelrf_task1_trained_tuned.joblib')
    y_true = y_train
    y_pred = y_pred_test
    save_test_results_to_csv(y_true,y_pred, "test_result_rf_1.csv")
    print(f"Training and testing time: {time_elapsed(time.time() - start_time)}")
    print(f"Training Accuracy: {accuracy_train}")
    print(f"Testing Accuracy: {accuracy_test}")


def task_1_gbc(save_type, model_file_path):
    """

    :return:
    """
    model_file = ''
    if save_type in ['t', 'tt']:
        model_file = model_file_path

    if save_type in ['t', 'tt'] and os.path.exists(model_file):
        print(f"Loading saved model from {model_file}")
        best_model = joblib.load(model_file)
        best_hyperparams=best_model.get_params()
    else:
        start_time = time.time()

        # Data preprocessing
        print("Data Preprocessing...")
        trainLoader, testLoader, X_valid, Y_valid, _, _ = data_preprocessing(model_name='GBC', task_name='Task 1',
                                                                             batchsize=151, temp_size=0.3,
                                                                             test_size=0.5)

        print("Tuning hyperparameters...")
        best_hyperparams, best_score = tune_gbc_hyperparameters(X_valid, Y_valid)
        print(f"Tuning time: {time_elapsed(time.time() - start_time)}")

        # Create and save the best model
        start_time = time.time()
        print("Creating and saving the best model...")
        best_model = GradientBoostingClassifier(**best_hyperparams)
        joblib.dump(best_model,'./entire_modelgbc_task1_tuned.joblib')  # Saving tuned model

    # Train and Test Gradient Boosting Descent Model
    start_time = time.time()
    print("Training and Testing Gradient Boosting Descent Model...")
    y_train, y_test, Y_pred_cv, y_pred_test, accuracy_train, accuracy_test, train_tuned_model = GBC_cv(trainLoader, testLoader, best_hyperparams)
    joblib.dump(train_tuned_model,'./entire_modelgbc_task1_trained_tuned.joblib')
    y_true = y_train
    y_pred = y_pred_test
    save_test_results_to_csv(y_true,y_pred, "test_result_gbc_1.csv")
    print(f"Training and testing time: {time_elapsed(time.time() - start_time)}")
    print(f"Training Accuracy: {accuracy_train}")
    print(f"Testing Accuracy: {accuracy_test}")


def task_1_cnn(save_type, model_file_path):
    """

    :return:
    """
    model_file = ''
    if save_type in ['t', 'tt']:
        model_file = model_file_path

    if save_type in ['t', 'tt'] and os.path.exists(model_file):
        print(f"Loading saved model from {model_file}")
        best_model = torch.load(model_file)

    start_time = time.time()

    # Data preprocessing
    print("Data Preprocessing...")
    trainLoader, testLoader, X_valid, Y_valid, output_size, input_size = data_preprocessing(model_name='CNN',
                                                                                            task_name='Task 1'
                                                                                            , batchsize=64,
                                                                                            temp_size=0.3,
                                                                                            test_size=0.5)
    print(f"Data preprocessing time: {time_elapsed(time.time() - start_time)}")
    if save_type == 'new':
    # Tune hyperparameters
        print(input_size)
        modelCNN = ConvNeuralNetwork(input_channels=input_size, kernel_size1=(3, 3, 3), kernel_size2=(3, 3, 3), output_size=output_size,
                                     hidden_size1=64,
                                     hidden_size2=128, hidden_size3=50, stride=1, padding=1)
        start_time = time.time()
        print("Tuning hyperparameters...")
        best_hyperparams, best_score = tune_cnn_hyperparameters(modelCNN, X_valid, Y_valid,output_size,max_epochs=50,cv=3)

        print(f"Tuning time: {time_elapsed(time.time() - start_time)}")

        # Create and save the best model
        start_time = time.time()
        print("Creating and saving the best model...")
        best_model = ConvNeuralNetwork(input_channels=3,
                                       kernel_size1=best_hyperparams['module__kernel_size1'],
                                       kernel_size2=best_hyperparams['module__kernel_size2'],
                                       activation_function=best_hyperparams['module__activation_function'],
                                       output_size=output_size,
                                       hidden_size1=best_hyperparams['module__hidden_size1'],
                                       hidden_size2=best_hyperparams['module__hidden_size2'],
                                       hidden_size3=best_hyperparams['module__hidden_size3'],
                                       stride=1,
                                       padding=1)
        torch.save(best_model, './entire_modelcnn_task1_tuned.pth')  # change path
        print(f"Model creation and saving time: {time_elapsed(time.time() - start_time)}")


    if save_type in ['new', 't']:
        # Train the best model
        start_time = time.time()
        print("Training the best model...")
        best_model, all_loss, all_accuracy, all_epoch = train_cnn_model(trainLoader, best_model, nbr_epoch=50)
        print(f"Training time: {time_elapsed(time.time() - start_time)}")
        save_training_results_to_csv(all_loss,all_accuracy,all_epoch,"training_result_cnn_1.csv")

        start_time = time.time()
        print("Saving tuned and trained model..")
        torch.save(best_model, './entire_modelcnn_task1_tuned_trained.pth')  # change path
        print(f"Model creation and saving time: {time_elapsed(time.time() - start_time)}")

    start_time = time.time()
    print("Testing tuned model...")
    y_true, y_pred = test_cnn_model(testLoader, best_model)
    save_test_results_to_csv(y_true,y_pred, "training_result_cnn_1.csv")
    print(f"Testing time: {time_elapsed(time.time() - start_time)}")


def task_2_nn(save_type, model_file_path):
    """

    :return:
    """
    model_file = ''
    if save_type in ['t', 'tt']:
        model_file = model_file_path

    if save_type in ['t', 'tt'] and os.path.exists(model_file):
        print(f"Loading saved model from {model_file}")
        best_model = torch.load(model_file)

    start_time = time.time()

    # Data preprocessing
    print("Data Preprocessing...")
    trainLoader, testLoader, X_valid, Y_valid, output_size, input_size = data_preprocessing(model_name='NN',
                                                                                            task_name='Task 2',
                                                                                            batchsize=151,
                                                                                            temp_size=0.3,
                                                                                            test_size=0.5)
    print(f"Data preprocessing time: {time_elapsed(time.time() - start_time)}")

    if save_type == 'new':
        # Tune hyperparameters
        modelNN = NeuralNetwork(output_size)
        start_time = time.time()
        print("Tuning hyperparameters...")
        best_hyperparams, best_score = tune_nn_hyperparameters(modelNN, X_valid, Y_valid, output_size, max_epochs=10)

        print(f"Tuning time: {time_elapsed(time.time() - start_time)}")

        # Create and save the best model
        start_time = time.time()
        print("Creating and saving the best model...")
        best_model = NeuralNetwork(output_size=output_size, input_size=input_size,
                                   hidden_size1=best_hyperparams['module__hidden_size1'],
                                   hidden_size2=best_hyperparams['module__hidden_size2'])
        torch.save(best_model, './entire_modelnn_task2_tuned.pth')  # change path
        print(f"Model creation and saving time: {time_elapsed(time.time() - start_time)}")

    if save_type in ['new', 't']:
        # Train the best model
        start_time = time.time()
        print("Training the best model...")
        best_model, all_loss, all_accuracy, all_epoch = train_nn_model(trainLoader, best_model, nbr_epoch=50)
        print(f"Training time: {time_elapsed(time.time() - start_time)}")
        save_training_results_to_csv(all_loss,all_accuracy,all_epoch,"training_result_nn_2.csv")

        start_time = time.time()
        print("Saving tuned and trained model..")
        torch.save(best_model, './entire_modelnn_task2_tuned_trained.pth')  # change path
        print(f"Model creation and saving time: {time_elapsed(time.time() - start_time)}")

    # Testing the tuned model
    start_time = time.time()
    print("Testing tuned model...")
    y_true, y_pred = test_nn_model(testLoader, best_model)
    save_test_results_to_csv(y_true,y_pred, "training_result_nn_2.csv")
    print(f"Testing time: {time_elapsed(time.time() - start_time)}")


def task_2_gru(save_type, model_file_path):
    """

    :return:
    """
    model_file = ''
    if save_type in ['t', 'tt']:
        model_file = model_file_path

    if save_type in ['t', 'tt'] and os.path.exists(model_file):
        print(f"Loading saved model from {model_file}")
        best_model = torch.load(model_file)

    start_time = time.time()

    # Data preprocessing
    print("Data Preprocessing...")
    trainLoader, testLoader, X_valid, Y_valid, output_size, input_size = data_preprocessing(model_name='GRU',
                                                                                            task_name='Task 2',
                                                                                            batchsize=151,
                                                                                            temp_size=0.3,
                                                                                            test_size=0.5)
    print(f"Data preprocessing time: {time_elapsed(time.time() - start_time)}")

    # Tune hyperparameters
    if save_type == 'new':
        modelGRU = GRUNeuralNetwork(input_size=input_size, output_size=output_size)

        start_time = time.time()
        print("Tuning hyperparameters...")
        best_hyperparams, best_score = tune_gru_hyperparameters(modelGRU, X_valid, Y_valid, max_epochs=5)

        print(f"Tuning time: {time_elapsed(time.time() - start_time)}")

        # Create and save the best model
        start_time = time.time()
        print("Creating and saving the best model...")
        best_model = GRUNeuralNetwork(output_size=output_size,
                                      hidden_size1=best_hyperparams['module__hidden_size1'],
                                      activation=best_hyperparams['module_activation'])
        torch.save(best_model, './entire_modelgru_task2_tuned.pth')  # change path
        print(f"Model creation and saving time: {time_elapsed(time.time() - start_time)}")

    if save_type in ['new', 't']:

        # Train the best model
        start_time = time.time()
        print("Training the best model...")
        best_model, all_loss, all_accuracy, all_epoch = train_gru_model(trainLoader, best_model, nbr_epoch=50)
        print(f"Training time: {time_elapsed(time.time() - start_time)}")

        save_training_results_to_csv(all_loss,all_accuracy,all_epoch,"training_result_gru_2.csv")

        start_time = time.time()
        print("Saving tuned and trained model...")
        torch.save(best_model, './entire_modelgru_task2_tuned_trained.pth')  # change path
        print(f"Model creation and saving time: {time_elapsed(time.time() - start_time)}")

    # Testing the tuned model
    start_time = time.time()
    print("Testing tuned model...")
    y_true, y_pred = test_gru_model(testLoader, best_model)
    save_test_results_to_csv(y_true,y_pred, "training_result_gru_2.csv")
    print(f"Testing time: {time_elapsed(time.time() - start_time)}")


def task_2_rf(save_type, model_file_path):
    """

    :return:
    """
    model_file = ''
    if save_type in ['t', 'tt']:
        model_file = model_file_path

    if save_type in ['t', 'tt'] and os.path.exists(model_file):
        print(f"Loading saved model from {model_file}")
        best_model = joblib.load(model_file)
        best_hyperparams=best_model.get_params()
    else:
        start_time = time.time()

        # Data preprocessing
        print("Data Preprocessing...")
        trainLoader, testLoader, X_valid, Y_valid, _, _ = data_preprocessing(model_name='RF', task_name='Task 2',
                                                                             batchsize=152, temp_size=0.3,
                                                                             test_size=0.5)

        print("Tuning hyperparameters...")
        best_hyperparams, best_score = tune_rf_hyperparameters(X_valid, Y_valid)
        print(f"Tuning time: {time_elapsed(time.time() - start_time)}")

        # Create and save the best model
        start_time = time.time()
        print("Creating and saving the best model...")
        best_model = RandomForestClassifier(**best_hyperparams)
        joblib.dump(best_model,'./entire_modelrf_task2_tuned.joblib')  # Saving tuned model

    # Train and Test RandomForest Model
    start_time = time.time()
    print("Training and Testing RandomForest Model...")
    y_train, y_test, Y_pred_cv, y_pred_test, accuracy_train, accuracy_test, train_tuned_model = RF_cv(trainLoader, testLoader, best_hyperparams)
    joblib.dump(train_tuned_model,'./entire_modelrf_task2_trained_tuned.joblib')
    y_true = y_train
    y_pred = y_pred_test
    save_test_results_to_csv(y_true,y_pred, "test_result_rf_2.csv")

    print(f"Training and testing time: {time_elapsed(time.time() - start_time)}")
    print(f"Training Accuracy: {accuracy_train}")
    print(f"Testing Accuracy: {accuracy_test}")


def task_2_gbc(save_type, model_file_path):
    """

    :return:
    """
    model_file = ''
    if save_type in ['t', 'tt']:
        model_file = model_file_path

    if save_type in ['t', 'tt'] and os.path.exists(model_file):
        print(f"Loading saved model from {model_file}")
        best_model = joblib.load(model_file)
        best_hyperparams=best_model.get_params()
    else:
        start_time = time.time()

        # Data preprocessing
        print("Data Preprocessing...")
        trainLoader, testLoader, X_valid, Y_valid, _, _ = data_preprocessing(model_name='GBC', task_name='Task 2',
                                                                             batchsize=151, temp_size=0.3,
                                                                             test_size=0.5)

        print("Tuning hyperparameters...")
        best_hyperparams, best_score = tune_gbc_hyperparameters(X_valid, Y_valid)
        print(f"Tuning time: {time_elapsed(time.time() - start_time)}")

        # Create and save the best model
        start_time = time.time()
        print("Creating and saving the best model...")
        best_model = GradientBoostingClassifier(**best_hyperparams)
        joblib.dump(best_model,'./entire_modelgbc_task2_tuned.joblib')  # Saving tuned model

    # Train and Test Gradient Boosting Descent Model
    start_time = time.time()
    print("Training and Testing Gradient Boosting Descent Model...")
    y_train, y_test, Y_pred_cv, y_pred_test, accuracy_train, accuracy_test, train_tuned_model = GBC_cv(trainLoader, testLoader, best_hyperparams)
    joblib.dump(train_tuned_model,'./entire_modelgbc_task2_trained_tuned.joblib')
    y_true = y_train
    y_pred = y_pred_test
    save_test_results_to_csv(y_true,y_pred, "test_result_gbc_2.csv")
    print(f"Training and testing time: {time_elapsed(time.time() - start_time)}")
    print(f"Training Accuracy: {accuracy_train}")
    print(f"Testing Accuracy: {accuracy_test}")


def task_2_cnn(save_type, model_file_path):
    """

    :return:
    """
    model_file = ''
    if save_type in ['t', 'tt']:
        model_file = model_file_path

    if save_type in ['t', 'tt'] and os.path.exists(model_file):
        print(f"Loading saved model from {model_file}")
        best_model = torch.load(model_file)

    start_time = time.time()

    # Data preprocessing
    print("Data Preprocessing...")
    trainLoader, testLoader, X_valid, Y_valid, output_size, input_size = data_preprocessing(model_name='CNN',
                                                                                            task_name='Task 2'
                                                                                            , batchsize=64,
                                                                                            temp_size=0.3,
                                                                                            test_size=0.5)
    print(f"Data preprocessing time: {time_elapsed(time.time() - start_time)}")
    if save_type == 'new':
        # Tune hyperparameters
        print(input_size)
        modelCNN = ConvNeuralNetwork(input_channels=input_size, kernel_size1=(3, 3, 3), kernel_size2=(3, 3, 3), output_size=output_size,
                                     hidden_size1=128,
                                     hidden_size2=256, hidden_size3=100, stride=1, padding=1)
        start_time = time.time()
        print("Tuning hyperparameters...")
        best_hyperparams, best_score = tune_cnn_hyperparameters(modelCNN, X_valid, Y_valid,output_size,max_epochs=50,cv=3)

        print(f"Tuning time: {time_elapsed(time.time() - start_time)}")

        # Create and save the best model
        start_time = time.time()
        print("Creating and saving the best model...")
        best_model = ConvNeuralNetwork(input_channels=3,
                                       kernel_size1=best_hyperparams['module__kernel_size1'],
                                       kernel_size2=best_hyperparams['module__kernel_size2'],
                                       activation_function=best_hyperparams['module__activation_function'],
                                       output_size=output_size,
                                       hidden_size1=best_hyperparams['module__hidden_size1'],
                                       hidden_size2=best_hyperparams['module__hidden_size2'],
                                       hidden_size3=best_hyperparams['module__hidden_size3'],
                                       stride=1,
                                       padding=1)
        torch.save(best_model, './entire_modelcnn_task2_tuned.pth')  # change path
        print(f"Model creation and saving time: {time_elapsed(time.time() - start_time)}")


    if save_type in ['new', 't']:
        # Train the best model
        start_time = time.time()
        print("Training the best model...")
        best_model, all_loss, all_accuracy, all_epoch = train_cnn_model(trainLoader, best_model, nbr_epoch=3)
        print(f"Training time: {time_elapsed(time.time() - start_time)}")
        save_training_results_to_csv(all_loss,all_accuracy,all_epoch,"training_result_cnn_2.csv")
        start_time = time.time()
        print("Saving tuned and trained model..")
        torch.save(best_model, './entire_modelcnn_task2_tuned_trained.pth')  # change path
        print(f"Model creation and saving time: {time_elapsed(time.time() - start_time)}")

    start_time = time.time()
    print("Testing tuned model...")
    y_true, y_pred = test_cnn_model(testLoader, best_model)
    save_test_results_to_csv(y_true,y_pred, "test_result_cnn_2.csv")
    print(f"Testing time: {time_elapsed(time.time() - start_time)}")


def run_task(task, model, save_type):
    """

    :param save_type:
    :param task:
    :param model:
    :return:
    """
    if task == 1:

        if model == 'NN':

            model_file_path = ''
            if save_type == 't':
                model_file_path = './entire_modelnn_task1_tuned.pth'
            elif save_type == 'tt':
                model_file_path = './entire_modelnn_task1_tuned_trained.pth'

            if model_file_path and not os.path.exists(model_file_path):
                print(f"Model file {model_file_path} does not exist. Proceeding with a new model.")
                save_type = 'new'

            task_1_nn(save_type, model_file_path)

        elif model == 'GRU':
            model_file_path = ''
            if save_type == 't':
                model_file_path = './entire_modelgru_task1_tuned.pth'
            elif save_type == 'tt':
                model_file_path = './entire_modelgru_task1_tuned_trained.pth'

            if model_file_path and not os.path.exists(model_file_path):
                print(f"Model file {model_file_path} does not exist. Proceeding with a new model.")
                save_type = 'new'

            task_1_gru(save_type, model_file_path)

        elif model == 'RF':

            model_file_path = ''
            if save_type == 't':
                model_file_path = './entire_modelrf_task1_tuned.joblib'
            elif save_type == 'tt':
                model_file_path = './entire_modelrf_task1_tuned_trained.joblib'

            if model_file_path and not os.path.exists(model_file_path):
                print(f"Model file {model_file_path} does not exist. Proceeding with a new model.")
                save_type = 'new'

            task_1_rf(save_type, model_file_path)

        elif model == 'GBC':

            model_file_path = ''
            if save_type == 't':
                model_file_path = './entire_modelgbc_task1_tuned.joblib'
            elif save_type == 'tt':
                model_file_path = './entire_modelgbc_task1_tuned_trained.joblib'

            if model_file_path and not os.path.exists(model_file_path):
                print(f"Model file {model_file_path} does not exist. Proceeding with a new model.")
                save_type = 'new'

            task_1_gbc(save_type, model_file_path)

        elif model == 'CNN':

            model_file_path = ''
            if save_type == 't':
                model_file_path = './entire_modelcnn_task1_tuned.pth'
            elif save_type == 'tt':
                model_file_path = './entire_modelcnn_task1_tuned_trained.pth'

            if model_file_path and not os.path.exists(model_file_path):
                print(f"Model file {model_file_path} does not exist. Proceeding with a new model.")
                save_type = 'new'

            task_1_cnn(save_type, model_file_path)

    elif task == 2:

        if model == 'NN':

            model_file_path = ''
            if save_type == 't':
                model_file_path = './entire_modelnn_task2_tuned.pth'
            elif save_type == 'tt':
                model_file_path = './entire_modelnn_task2_tuned_trained.pth'

            if model_file_path and not os.path.exists(model_file_path):
                print(f"Model file {model_file_path} does not exist. Proceeding with a new model.")
                save_type = 'new'

            task_2_nn(save_type, model_file_path)

        elif model == 'GRU':

            model_file_path = ''
            if save_type == 't':
                model_file_path = './entire_modelgru_task2_tuned.pth'
            elif save_type == 'tt':
                model_file_path = './entire_modelgru_task2_tuned_trained.pth'

            if model_file_path and not os.path.exists(model_file_path):
                print(f"Model file {model_file_path} does not exist. Proceeding with a new model.")
                save_type = 'new'

            task_2_gru(save_type, model_file_path)

        elif model == 'RF':

            model_file_path = ''
            if save_type == 't':
                model_file_path = './entire_modelrf_task2_tuned.joblib'
            elif save_type == 'tt':
                model_file_path = './entire_modelrf_task2_tuned_trained.joblib'

            if model_file_path and not os.path.exists(model_file_path):
                print(f"Model file {model_file_path} does not exist. Proceeding with a new model.")
                save_type = 'new'

            task_2_rf(save_type, model_file_path)

        elif model == 'GBC':

            model_file_path = ''
            if save_type == 't':
                model_file_path = './entire_modelgbc_task2_tuned.joblib'
            elif save_type == 'tt':
                model_file_path = './entire_modelgbc_task2_tuned_trained.joblib'

            if model_file_path and not os.path.exists(model_file_path):
                print(f"Model file {model_file_path} does not exist. Proceeding with a new model.")
                save_type = 'new'

            task_2_gbc(save_type, model_file_path)

        elif model == 'CNN':

            model_file_path = ''
            if save_type == 't':
                model_file_path = './entire_modelcnn_task2_tuned.pth'
            elif save_type == 'tt':
                model_file_path = './entire_modelcnn_task2_tuned_trained.pth'

            if model_file_path and not os.path.exists(model_file_path):
                print(f"Model file {model_file_path} does not exist. Proceeding with a new model.")
                save_type = 'new'

            task_2_cnn(save_type, model_file_path)


if __name__ == "__main__":
    while True:
        try:
            print("Is AMD GPU available:", torch.cuda.is_available())
            print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")
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

                while True:
                    save_type_input = input("Do you want to use a saved model? Enter 'yes' or 'no'. If you have "
                                            "already existing model files 'no' will create new ones and destroy the "
                                            "existing ones (If you want to keep the already existing ones change the "
                                            "names of the existing files): ")

                    if save_type_input.lower() == 'yes':
                        save_type_choice = input("Choose the save type of the model you want to use: 't' for model "
                                                 "tuned with the best parameters or 'tt' for model tuned  and trained "
                                                 "with best parameters: ")

                        if save_type_choice.lower() not in ['t', 'tt']:
                            print("Invalid saved model choice. Please enter a valid choice.")
                            continue
                        elif save_type_choice == 't':
                            save_type = 't'
                        else:
                            save_type = 'tt'

                    elif save_type_input.lower() == 'no':
                        save_type = 'new'
                    else:
                        print("Invalid response type 'yes' or 'no'")
                        continue
                    if save_type not in ['t', 'tt', 'no', 'new']:
                        print('sd')
                    else:
                        run_task(task, model_input, save_type)
                    break
            break


        except ValueError:
            print("An error occurred.")
        except Exception as e:
            # Handle other potential exceptions
            print(f"An error occurred: {e}")
        except KeyboardInterrupt:
            print("\nUser interrupted the program. Exiting...")
            break
    # List to hold the names of the relevant CSV files
    csv_training_files = []
    csv_test_files = []
    # Directory where the files are located (assuming current directory)
    directory = '.'

    # Loop through all files in the directory
    for file in os.listdir(directory):
        if file.startswith('training_result_') and file.endswith('.csv'):
            csv_training_files.append(os.path.join(directory, file))

    # Check if we found any files
    if csv_training_files:
        # Make sure you have imported plot_accuracy_for_networks from visualization.py
        plot_accuracy_for_networks(csv_training_files)
    else:
        print("No training result CSV files found.")

    for file in os.listdir(directory):
        if file.startswith('test_result_') and file.endswith('.csv'):
            csv_test_files.append(os.path.join(directory, file))

    # Check if we found any files
    if csv_test_files:
        # Make sure you have imported plot_accuracy_for_networks from visualization.py
        plot_and_save_confusion_matrix(csv_test_files)
    else:
        print("No test result CSV files found.")
#%%
