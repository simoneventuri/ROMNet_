import numpy as np


class TrainState(object):
    
    def __init__(self):

        self.epoch           = 0
        self.step            = 0
 
        # Current data 
        self.X_train         = None
        self.y_train         = None
        self.train_aux_vars  = None
        self.X_test          = None
        self.y_test          = None
        self.test_aux_vars   = None

        # Results of current step
        # Train results
        self.loss_train      = None
        self.y_pred_train    = None
        # Test results 
        self.loss_test       = None
        self.y_pred_test     = None
        self.y_std_test      = None
        self.metrics_test    = None

        # The best results correspond to the min train loss
        self.best_step       = 0
        self.best_loss_train = np.inf
        self.best_loss_test  = np.inf
        self.best_y          = None
        self.best_ystd       = None
        self.best_metrics    = None



    #=======================================================================================================================================
    def set_data_train(self, X_train, y_train, train_aux_vars=None):
        
        self.X_train = X_train
        self.y_train = y_train
        self.train_aux_vars = train_aux_vars

    #=======================================================================================================================================



    #=======================================================================================================================================
    def set_data_test(self, X_test, y_test, test_aux_vars=None):

        self.X_test = X_test
        self.y_test = y_test
        self.test_aux_vars = test_aux_vars

    #=======================================================================================================================================



    #=======================================================================================================================================
    def update_best(self):

        if self.best_loss_train > np.sum(self.loss_train):
            self.best_step = self.step
            self.best_loss_train = np.sum(self.loss_train)
            self.best_loss_test = np.sum(self.loss_test)
            self.best_y = self.y_pred_test
            self.best_ystd = self.y_std_test
            self.best_metrics = self.metrics_test

    #=======================================================================================================================================



    #=======================================================================================================================================
    def disregard_best(self):

        self.best_loss_train = np.inf

    #=======================================================================================================================================



    #=======================================================================================================================================
    def packed_data(self):
        
        def merge_values(values):
            if values is None:
                return None
            return np.hstack(values) if isinstance(values, (list, tuple)) else values

        X_train   = merge_values(self.X_train)
        y_train   = merge_values(self.y_train)
        X_test    = merge_values(self.X_test)
        y_test    = merge_values(self.y_test)
        best_y    = merge_values(self.best_y)
        best_ystd = merge_values(self.best_ystd)

        return X_train, y_train, X_test, y_test, best_y, best_ystd

    #=======================================================================================================================================
