import numpy as np
import matplotlib.pyplot as plt  # for 畫圖用
# import matplotlib.image as mpimg
from PIL import Image
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


def readTrain():
    train = pd.read_csv("./typhoon06/2012su.csv")
    return train


def normalization(training_set):
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    
    return training_set_scaled



def buildTrain(train, pastDay=30, futureDay=5):
    X_train, Y_train = [], []
    
    train = pd.DataFrame(train, columns=['h', 'flow', 'sand'])
    
    for i in range(train.shape[0]-futureDay-pastDay): # range(train.shape[0]-futureDay-pastDay)
        X_train.append(np.array(train.iloc[i:i+pastDay]))
        # Y_train.append(list(map(list, [np.array(train.iloc[i+pastDay:i+pastDay+futureDay]["sand"])])))
        
        ans = []
        for data in train.iloc[i+pastDay:i+pastDay+futureDay]["sand"]:
            temp = []
            temp.append(data)
            ans.append(temp)
            
        Y_train.append(ans)
    
    return np.array(X_train), np.array(Y_train)


def buildLSTM(x_train, y_train):
    regressor = Sequential()
    
    # Adding the first LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 3)))
    regressor.add(Dropout(0.2))

    # Adding a second LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))

    # Adding a third LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))

    # Adding a fourth LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50))
    regressor.add(Dropout(0.2))
    
    regressor.add(Dense(units = 1))
    
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    regressor.fit(x_train, y_train, epochs = 1500, batch_size = 64)
    
    dataset_test = pd.read_csv("./typhoon06/2012su.csv")
    real_sand = dataset_test.iloc[0:80]["sand"]
    
    dataset_test = normalization(dataset_test)
    dataset_test = pd.DataFrame(dataset_test, columns=['h', 'flow', 'sand'])
    
    X_test = []
    for i in range(dataset_test.shape[0]-3):
        X_test.append(np.array(dataset_test.iloc[i:i+3]))
    
    # X_test, Y_test = buildTrain(normalization(dataset_test), 3, 1, 0)
    X_test = np.array(X_test)
    # print(X_test)
    # print(X_test.shape)
    
    sc_Y_test = MinMaxScaler(feature_range = (0, 1)).fit(np.array(real_sand).reshape(-1, 1))
    
    predicted_sand = regressor.predict(X_test)
    # print(type(predicted_sand))
    # print(predicted_sand.shape)
    
    # print(predicted_sand)
    
    predicted_sand = pd.DataFrame(predicted_sand)
    predicted_sand = sc_Y_test.inverse_transform(predicted_sand)
    
    plt.plot(real_sand, color = 'red', label = 'Real sand')
    plt.plot(predicted_sand, color = 'blue', label = 'Predicted sand')
    plt.title('sand Prediction')
    plt.xlabel('Time')
    plt.ylabel('sand')
    plt.legend()
    # plt.show()
    plt.savefig('./result_image/sand-20221129.png', dpi=1000)
    
    image = Image.open('./result_image/sand-20221129.png')
    image.show()



if __name__ == "__main__":
    training_set = readTrain()
    
    training_set_scaled = normalization(training_set)
    
    x_train, y_train = buildTrain(training_set_scaled, 3, 1)
    # print(x_train)
    # print(y_train)
    # print(x_train.shape)
    # print(y_train.shape)
    # (73, 3, 3)
    # (73, 1, 1)
    
    # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    buildLSTM(x_train, y_train)