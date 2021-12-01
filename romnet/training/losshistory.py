
import matplotlib
#matplotlib.use('TkAgg')
from matplotlib                           import pyplot as plt



class LossHistory(object):

    def __init__(self):
        self.steps        = []
        self.loss_train   = []
        self.loss_test    = []
        self.metrics_test = []
        self.loss_weights = 1


    #=======================================================================================================================================
    def set_loss_weights(self, loss_weights):
        
        self.loss_weights = loss_weights

    #=======================================================================================================================================



    #=======================================================================================================================================
    def append(self, step, loss_train, loss_test, metrics_test):
        
        self.steps.append(step)

        self.loss_train.append(loss_train)
        
        if loss_test is None:
            loss_test = self.loss_test[-1]
        self.loss_test.append(loss_test)

        if metrics_test is None:
            metrics_test = self.metrics_test[-1]
        self.metrics_test.append(metrics_test)

    #=======================================================================================================================================



    #=======================================================================================================================================
    def plot(self, InputData):

        if (self.loss_train is not None):
            fig = plt.figure()
            plt.plot(self.loss_train, label='loss')
            plt.plot(self.loss_test,  label='val_loss')
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