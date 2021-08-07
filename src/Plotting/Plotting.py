import os
import sys
import matplotlib
#matplotlib.use('TkAgg')
from matplotlib                           import pyplot as plt
import math
import tensorflow                             as tf
import numpy                                  as np
from matplotlib                           import pyplot as plt
from matplotlib                           import animation


#=======================================================================================================================================
def plot_losseshistory(InputData, history):

    fig = plt.figure()
    plt.plot(history.history['loss'],     label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.title('Training History')
    FigPath = InputData.PathToFigFld + '/LossesHistory.png'
    fig.savefig(FigPath, dpi=1000)
    #plt.show()
    plt.close()

#=======================================================================================================================================
