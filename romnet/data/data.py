

class Data(object):

    def __init__(self, InputData, system):

        self.Type = InputData.DataType


    #===========================================================================
    def res_fn(self, net):
        '''Residual loss function'''

        self.NVarsx = net.NVarsx
        self.NVarsy = net.NVarsy

        def residual(inputs, training=True):
            return None
        
        return residual
    #===========================================================================
