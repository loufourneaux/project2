import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA

def data_preprocessing(model_name, task_name, batchsize, temp_size = 0.2, test_size =0.5, num_channels =3):
    """
    
    :param model_name: 
    :param batchsize: 
    :param temp_size: 
    :param test_size: 
    :param num_channels: 
    :return: 
    """
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    #change path
    df = pd.read_parquet(r"/Users/loufourneaux/Desktop/EPFL/MA1/ML/project2/All_Relative_Results_Cleaned.parquet")

    df_clean = df.dropna()
    index = df_clean.columns.get_loc('time(s)')
    
    if task_name == 'Task 1':        
        Y = df_clean['Exercise']
    elif task_name == 'Task 2':
        #Fusion of the column exercise and Set to know the mistake 
        Y = df_clean['Exercise']+ 'avec erreur'+ df_clean['Set'] 
    label_encoder = LabelEncoder()
    Y_encoded = label_encoder.fit_transform(Y)

    if model_name == 'NN' or model_name == 'CNN':
        index +=1

    X = df_clean.iloc[:, index:]

    if model_name == 'NN':
        X_tensor = torch.tensor(X.values, dtype=torch.float32)

    elif model_name =='CNN':
        cols_x = [col for col in X.columns if col.endswith('x')]
        cols_y = [col for col in X.columns if col.endswith('y')]
        cols_z = [col for col in X.columns if col.endswith('z')]
        tensor_5D = torch.zeros((len(X), 3, 33, 1, 1), dtype=torch.float32)
        nbr_of_points = (X.shape[1])//3
        X_reshape = X.values.reshape(-1,nbr_of_points,3)
        tensor_4D_transposed = np.expand_dims(np.expand_dims(np.transpose(X_reshape, (0, 2, 1)), axis=3), axis=4)
        tensor_5D = torch.from_numpy(tensor_4D_transposed)
        tensor_5D = tensor_5D.permute(0, 2, 1, 3, 4)

        data_x = X[cols_x].values
        data_y = X[cols_y].values
        data_z = X[cols_z].values

        mean_x = np.mean(data_x, axis=0)
        mean_y = np.mean(data_y, axis=0)
        mean_z = np.mean(data_z, axis=0)

        mean_coords = np.stack([mean_x, mean_y, mean_z], axis=1)
        pca = PCA(n_components=3)
        reduced_coords = pca.fit_transform(mean_coords)


        normalized_coords = (reduced_coords - reduced_coords.min(0)) / reduced_coords.ptp(0)
        print(normalized_coords.shape)
        grid_coords = np.round(normalized_coords * np.array([2, 2, 10])).astype(int)
        depth, height, width = 3, 3, 11

        num_channels = num_channels

        num_samples = 2183485
        grid_tensor = torch.zeros((num_samples, num_channels, depth, height, width))
        for feature_idx in range(33):  # Supposons que vous avez 33 features
            d, h, w = grid_coords[feature_idx]

            d, h, w = np.clip([d, h, w], 0, [depth - 1, height - 1, width - 1])

            grid_tensor[:, 0, d, h, w] = torch.tensor(X[cols_x[feature_idx]].values)
            grid_tensor[:, 1, d, h, w] = torch.tensor(X[cols_y[feature_idx]].values)
            grid_tensor[:, 2, d, h, w] = torch.tensor(X[cols_z[feature_idx]].values)

            X_tensor = grid_tensor

    Y_tensor = torch.tensor(Y_encoded, dtype=torch.long)

    X_train, X_temp, Y_train, Y_temp = train_test_split(X_tensor, Y_tensor, test_size=temp_size)
    X_test, X_validation, Y_test, Y_validation = train_test_split(X_temp, Y_temp, test_size=test_size)


    train_dataset = TensorDataset(X_train,Y_train)
    #validation_dataset = TensorDataset(X_validation,Y_validation)
    test_dataset = TensorDataset(X_test,Y_test)
    trainLoader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    #validationLoader = DataLoader(validation_dataset,batch_size=batchsize,shuffle=True)
    testLoader = DataLoader(test_dataset, batch_size=batchsize , shuffle=True)

    return trainLoader,testLoader, X_validation, Y_validation