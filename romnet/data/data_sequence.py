import numpy      as np
import pandas     as pd 
import tensorflow as tf
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.utils.DataSequence')
class DataSequence(tf.keras.utils.Sequence):

    #===========================================================================
    def __init__(
        self, 
        data, 
        data_ids, 
        batch_size = 64, 
        shuffle    = True
        ):
        ''' '''
        super(DataSequence, self).__init__()

        #===========================================================================
        def get_cardinality(x):
            if isinstance(x, (list,tuple)):
                return get_cardinality(x[0])
            else:
                card = x.shape[0]
                return card

        #===========================================================================


        # Whether to shuffle data when rebatching
        self.name     = self.__class__.__name__
        self.shuffle  = shuffle                                                 
        self.data_ids = list(data_ids)

        #-----------------------------------------------------------------------
        # Making sure that batch_size is a dictionary. If it is an integer,
        # make it a dictionary by repeating it for each data_id 
        if not isinstance(batch_size, (dict, int)):                             
            raise ValueError("`batch_size` has to be a `int` or `dict`.")       
        if isinstance(batch_size, int):
            self.batch_size = {data_id: batch_size for data_id in self.data_ids}
        else:
            self.batch_size = batch_size

        #-----------------------------------------------------------------------


        #-----------------------------------------------------------------------
        # Making sure that data is a dictionary. If it is a dictionary:
        #    - Removing the data_id that has empty data
        #    - Make sure that each value is a list: {'pts': (x,y), ...}
        #    - Computing cardinality of data (n_samples) for each data_id
        #    - Setting to 32 the batch_size of data_id that has no batch_size
        if not isinstance(data, dict):                                          
            raise ValueError("`data` has to be a `dict`.")                      
        else:                                                                   
            self.data, self.n_samples = {}, {}                                  
            for data_id in list(data_ids):            

                if data_id not in data or data[data_id] is None:
                    # Delete the corresponding `key` entry
                    self.data_ids.remove(data_id)
                    if data_id in self.batch_size:
                        self.batch_size.pop(data_id)
                else:
                    if not isinstance(data[data_id], (list,tuple)) and len(data[data_id]) > 2:
                        raise ValueError("Values passed into `data` dictionary"\
                                         " must be a `list`/`tuple` of "\
                                         "`np.ndarray` of length 2: "\
                                         "(inputs, outputs)")
                    else:
                        self.data[data_id]      = data[data_id]
                        self.n_samples[data_id] = get_cardinality(data[data_id])
                        if data_id not in self.batch_size:
                            self.batch_size[data_id] = 32       

        #-----------------------------------------------------------------------


        #-----------------------------------------------------------------------
        # Computing no of batches for each data_id
        self.steps = {data_id: int(np.ceil(self.n_samples[data_id] /           \
                      self.batch_size[data_id])) for data_id in self.data_ids}
        #-----------------------------------------------------------------------            

        
        self.on_epoch_end()

    #===========================================================================
    


    #===========================================================================
    def __len__(self):
        """Number of batches in the Sequence.
        Returns:
            The number of batches in the Sequence.
        """
        return max(list(self.steps.values()))

    #===========================================================================



    #===========================================================================
    @tf.function
    def __getitem__(self, index):
        """Gets batch at position `index`.
        Args:
            index: position of the batch in the Sequence.
        Returns:
            A batch
        """
        x, y, i = [ xyi for xyi in zip(*self.sequence[index]) ]
        return x, y, i

    #===========================================================================



    #===========================================================================
    @tf.function                                                                
    def on_epoch_end(self):                                                     
        """Method called at the end of every epoch.                                                                     
        """

        # At the end of the Epoch, possibly reshuffle each dataset, rebatch it, 
        # and zip it
        batches = dict()
        indexes = dict()
        for data_id, data_set in self.data.items():
 
            if self.shuffle:
                data_set = self.shuffle_data(data_set, data_id)

            batches[data_id] = self.batch_data(data_set, data_id)

        self.sequence = [ batch for batch in zip(*list(batches.values())) ]

    #===========================================================================



    #===========================================================================
    def shuffle_data(self, data_set, data_id):      

        # Generate random positions via np.random.permutation, 
        # and reassign components
        index     = np.random.permutation(self.n_samples[data_id])
        _data_set = []
        for xyi in data_set:

            if isinstance(xyi, (list,tuple)):
                # Multiple I/O
                if isinstance(xyi[0], (list,tuple)):
                    _data_set.append([i[index] for i in xyi])
                elif isinstance(xyi[0], pd.DataFrame):
                    _data_set.append([i.iloc[index] for i in xyi])                
            
            else:
                # Single I/O
                if isinstance(xyi, np.ndarray):
                    _data_set.append(xyi[index])
                elif isinstance(xyi, pd.DataFrame):
                    _data_set.append(xyi.iloc[index])
        
        return _data_set

    #===========================================================================



    #===========================================================================
    def batch_data(self, data_set, data_id):

        repeat    = int(np.ceil(self.__len__()/self.steps[data_id]))
        _data_set = []
        for xyi in data_set:

            if isinstance(xyi, (list,tuple)):
                # Multiple I/O
                spl  = [np.array_split(i, self.steps[data_id])*repeat for i in xyi]
                _xyi = [spl_i for spl_i in zip(*spl)]
        
            else:
                # Single I/O
                _xyi = np.array_split(xyi, self.steps[data_id])*repeat

            _data_set.append(_xyi)
        
        return [ _dset_i for _dset_i in zip(*_data_set) ]

    #===========================================================================
