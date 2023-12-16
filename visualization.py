import matplotlib.pyplot as plt
import csv
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import numpy as np
def save_training_results_to_csv(all_loss, all_accuracy, all_epoch, filename="training_results.csv"):
    """
    Saves the training results to a CSV file.

    :param all_loss: List of loss values for each epoch.
    :param all_accuracy: List of accuracy values for each epoch.
    :param all_epoch: List of epoch numbers.
    :param filename: Name of the file to save the results.
    """
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Loss", "Accuracy"])  # Writing the headers
        for epoch, loss, accuracy in zip(all_epoch, all_loss, all_accuracy):
            writer.writerow([epoch, loss, accuracy])

    print(f"Training results saved to {filename}")


def save_test_results_to_csv(y_true, y_pred, filename="test_results.csv" ):
    """
    Saves the test results (true and predicted values) to a CSV file.

    :param y_true: List of true labels.
    :param y_pred: List of predicted labels.
    :param filename: Name of the file to save the results.
    """
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["True Label", "Predicted Label"])  # Writing the headers
        for true, pred in zip(y_true, y_pred):
            writer.writerow([true, pred])

    print(f"Test results saved to {filename}")


def plot_accuracy_for_networks(csv_files, title="Accuracy over Epochs for Multiple Networks", save_path="accuracy_plot.png"):
    """
    Plots the accuracy over epochs for multiple neural networks with different colors and point styles.

    :param csv_files: List of CSV file paths, each containing epoch, loss, and accuracy data for a network.
    :param title: Title of the plot.
    :param save_path: Path where the plot image will be saved.
    """
    plt.figure(figsize=(10, 6))

    # Define colors for tasks and markers for network types
    task_colors = {'1': 'blue', '2': 'green'}
    network_markers = {
        'nn': 'o',   # Circle
        'gru': 's',  # Square
        'rf': '^',   # Triangle up
        'gbc': 'D',  # Diamond
        'cnn': 'p',  # Pentagon
    }

    for file in csv_files:
        data = pd.read_csv(file)
        if 'Epoch' in data.columns and 'Accuracy' in data.columns:
            # Extract model name and task number from the file name
            parts = os.path.splitext(os.path.basename(file))[0].split('_')
            model_name = parts[2].lower()  # Convert to lower case to match keys in network_markers
            task_number = parts[3]

            color = task_colors.get(task_number, 'black')  # Default to black if task not found
            marker = network_markers.get(model_name, '.')   # Default to point if network type not found

            # Plot with the specified color and marker
            plt.plot(data['Epoch'], data['Accuracy'], label=f"{model_name.upper()} Task {task_number}", color=color, marker=marker)
        else:
            print(f"Warning: The file {file} does not contain 'Epoch' and 'Accuracy' columns.")

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

    print(f"Plot saved to {save_path}")


def plot_and_save_confusion_matrix(csv_files, title="Confusion Matrices", save_path="confusion_matrices.png"):
    """
    Plots and saves a single plot containing all the confusion matrices for the specified test result CSV files.

    :param csv_files: List of CSV file paths, each containing true and predicted labels.
    :param title: Title for the overall plot of confusion matrices.
    :param save_path: Path where the combined plot image will be saved.
    """
    # Number of rows and columns for subplots
    n = len(csv_files)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))

    plt.figure(figsize=(ncols * 5, nrows * 4))

    for i, file in enumerate(csv_files, start=1):
        data = pd.read_csv(file)
        y_true = data['True Label']
        y_pred = data['Predicted Label']

        # Extract model name and task number from the file name
        parts = file.replace('test_results_', '').split('_')
        model_name = parts[2].lower()  # Convert to lower case to match keys in network_markers
        task_number = parts[3]


        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        cm_df = pd.DataFrame(cm)
        cm_data_save_path = f"{model_name}_task{task_number}_raw_data.csv"
        cm_df.to_csv(cm_data_save_path, index=False)

        # Create subplot
        plt.subplot(nrows, ncols, i)
        sns.heatmap(cm, annot=False)  # Disable text annotations
        plt.title(f'{model_name.upper()} Task {task_number}')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')

    # Adjust layout
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.9)

    # Save the combined plot
    plt.savefig(save_path)
    plt.close()

    print(f"Combined confusion matrix plot saved to {save_path}")
    print(f"Confusion matrix data saved to modelname_tasknumber_raw_data.csv")
