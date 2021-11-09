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


class LossHistory:

    def __init__(self, loss, val_loss):

        self.loss     = loss
        self.val_loss = val_loss


    #=======================================================================================================================================
    def plot(self, InputData):

        if (self.loss is not None):
            fig = plt.figure()
            plt.plot(self.loss,     label='loss')
            plt.plot(self.val_loss, label='val_loss')
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