#
# HW3 | Diamond Price Regression Keras
#
# This neural network is made to take input in the 
# form of data about diamonds in relation to their price
# and then predict the price of any given diamond based on
# its' values.
#
# Jake Roman & Jose Venecia
# 
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Input, Dense
from keras._tf_keras.keras.optimizers import Adam, SGD
from keras._tf_keras.keras.callbacks import EarlyStopping
import keras._tf_keras.keras.utils
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb

import argparse
from sklearn.model_selection import train_test_split


# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument("-e","--epochs",type=int,help="select how many epochs to train for",default=1000)
parser.add_argument("-b","--batchsize",type=int,help="set batch size for training",default=250)
parser.add_argument("-lr","--learningrate",type=float,help="set learning rate for training",default=0.001)
parser.add_argument("-df","--datafile",type=str,help="choose a custom dataset to use for training & testing",default="DiamondsPrices.csv")
parser.add_argument("-cm","--cachemodel",type=str,help="choose a .pt file to cache model weights",default=False)
args = parser.parse_args()
ROOT = os.path.dirname(os.path.abspath(__file__))  # root directory of this code
DATAFILE = 'DiamondsPrices.csv'

def main():
    
    
	# Load data
    data = np.loadtxt(os.path.join(ROOT, DATAFILE), delimiter=',', dtype=str)[1:]
    # set price max for use throughout the code, this is the upperbounds we allow the ai to guess
    global price_max
    price_max = np.max(np.int32(data[:, 6]))

    # x is the inputs (information about the diamonds)
    # t contains the actual prices of the diamonds
    x = [transform_data(sample) for sample in data] # Inputs
    t = [transform_label(sample) for sample in data] # Target Labels

    x = np.array(x)
    t = np.array(t)

    # both x and t are split into training and testing sets
    # shuffles by default
    x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.33, random_state=42)

    # Architecture of model
    model = Sequential()
    model.add(Input(shape=(9,)))
    model.add(Dense(units=10, activation='relu', name='hidden1'))
    model.add(Dense(units=25, activation='relu', name='hidden2'))
    model.add(Dense(units=50, activation='relu', name='hidden3'))
    model.add(Dense(units=25, activation='relu', name='hidden4'))
    model.add(Dense(units=1, activation='sigmoid', name='output'))
    model.summary()
    input("Press <Enter> to train this network...")

    # Select loss, optimizer, and metrics equations
    model.compile(
        loss='mean_absolute_percentage_error',           # measure of error
        optimizer=Adam(learning_rate=args.learningrate), # amount of change each batch
        metrics=['mean_absolute_percentage_error'])      # measure of success! (Lower number is better)
        
    # Ends training early if there is not much improvement in the model
    callback = EarlyStopping(
        monitor='loss',
        min_delta=1e-10,
        patience=20,
        verbose=1)
    
    # Train the network
    history = model.fit(x_train, t_train,
        epochs=args.epochs,
        batch_size=args.batchsize,
        callbacks=[callback],
        verbose=1,
        validation_data= (x_test, t_test))

    # Test the network grab error and metrics for printing
    metrics = model.evaluate(x_test, t_test, verbose=0)
    print(f'MAPE (Mean Absolute Percentage Error) = {metrics[1]:0.4f}')

    # Predict for comparison to actual prices on graph
    y = model.predict(x_test, verbose=1)

    # All the stuff to make the graph
    plt.figure(1)
    plt.scatter(price_inverse_transformation(t_test), price_inverse_transformation(y), s=1, alpha=0.3)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.xlim(-100,price_max)
    plt.ylim(-100,price_max)
    plt.plot([-100, price_max], [-100, price_max], color='black', linestyle='--', alpha=0.5)

    # Plot History
    plt.figure(2)
    plt.plot(history.history['loss'], label='Training', color='blue', alpha= 0.75)
    plt.plot(history.history['val_loss'], label='Testing', color='orange', alpha= 0.75)
    plt.title('Percentage Error Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, 50)
    plt.legend()
    plt.show()


def transform_data(data):
    # Extract and transform rows to be able to input into model
    return [
        float(data[0]),                                                                 # carats
        (["Fair","Good","Very Good","Premium","Ideal"].index(data[1])/4),               # transform cut into a 0-1 scale (higher is better)
        (["J","I","H","G","F","E","D"].index(data[2])/6),                               # transform color into a 0-1 scale (higher is better)
        (["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"].index(data[3])/7),           # transform clarity into a 0-1 scale (higher is better)
        ((float(data[4]) - 43)/36),                                                     # depth
        ((float(data[5]) - 43)/52),                                                     # table
        (float(data[7])/6),                                                             # size x
        (float(data[8])/6),                                                             # size y
        (float(data[9])/6),                                                             # size z
    ]

# Make the price a value between 0 and 1 to have an easier time getting the model to predict values
def transform_label(data):
    return int(data[6])/price_max

# Revert the price for the scatterplot
def price_inverse_transformation(data):
    # Reverses the process performed by transform_label
    return np.multiply(data,price_max)


    

if __name__ == "__main__":
	main()
