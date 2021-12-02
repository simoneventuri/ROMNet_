import pandas          as pd
import numpy           as np
import abc
import os 
from pyDOE         import lhs
import joblib          as jl
from tqdm          import tqdm
from scipy.spatial import KDTree

import tensorflow       as tf

from .data          import Data
from ..utils        import run_if_any_none
from .data_sequence import DataSequence
from ..pinn         import gradient



class PDE(Data):

    #===========================================================================
    def __init__(self, InputData, system):
        super(PDE, self).__init__(InputData, system)

        self.Type                = InputData.DataType

        self.PathToDataFld       = InputData.PathToDataFld

        self.SurrogateType       = InputData.SurrogateType
        if (self.SurrogateType == 'DeepONet'):
            self.BranchVars      = InputData.BranchVars
            self.TrunkVars       = InputData.TrunkVars
            self.InputVars       = self.BranchVars + self.TrunkVars
        else:
            self.InputVars       = InputData.InputVars
        try:    
            self.OutputVars      = system.OutputVars
        except:    
            self.OutputVars      = InputData.OutputVars
    
        self.NData               = 0
        self.xtrain, self.ytrain = None, None
        self.xtest,  self.ytest  = None, None

        try:
            self.TransFun        = InputData.TransFun
        except:
            self.TransFun        = None

        self.system              = system
        self.other_idxs          = system.other_idxs
        self.ind_idxs            = system.ind_idxs
        self.size_splits         = [len(self.other_idxs)]+[1]*len(self.ind_idxs)
        self.order               = system.order[0]
        self.get_residual        = system.get_residual
        if (system.fROM_anti):
            self.fROM_anti       = system.fROM_anti()
        else:
            self.fROM_anti       = None
        self.grad_fn             = gradient.get_grad_fn(self.order)
    
    #===========================================================================



    #===========================================================================
    # Generating Data 
    def get(self, InputData):
        print('\n[ROMNet]:   Generating Data')



        #-----------------------------------------------------------------------
        def initial_cond(data_obj):
            print('\n[ROMNet]:   Generating initial conditions ...')

            n_tot_list = [data_obj.n_train['ics'], data_obj.n_valid['ics'], data_obj.n_test]
            d          = len(data_obj.other_ranges)
            
            ic_list  = []
            for i_type in range(3):
                n_now = n_tot_list[i_type]

                if (n_now < 1):
                    ic_list.append(None)

                else:
                    # Contructing Design Matrix ----------------------------------------
                    X  = lhs(len(data_obj.other_ranges), samples=n_now)
                    X  = np.hstack(tuple(map(lambda i,x: X[:,i:i+1] * (x[1] - x[0]) + x[0], range(d), data_obj.other_ranges.values() )))
                    ic_list.append(X)

            return ic_list

        #-----------------------------------------------------------------------



        #-----------------------------------------------------------------------
        def split_idxs(n_tot, n_test):
            '''Generating Indexes for Splitting Data'''
            
            idxs_test  = np.random.choice(n_tot, n_test, replace=False)
            idxs_train = np.setdiff1d(np.arange(n_tot), idxs_test)
            assert np.intersect1d(idxs_train, idxs_test).size == 0
            
            return idxs_train, idxs_test

        #-----------------------------------------------------------------------



        #-----------------------------------------------------------------------
        def read_training_data(data_obj):
            print('\n[ROMNet]:   Reading training pts ...')

            data          = {}
            data['train'] = {}
            data['valid'] = {}

            for data_type in data:
                for data_id in data_obj.data_ids:

                    x_df   = pd.read_csv(data_obj.PathToDataFld+'/'+data_type+"/"+data_id+'/Input.csv')[data_obj.InputVars]
                    y_df   = pd.read_csv(data_obj.PathToDataFld+'/'+data_type+"/"+data_id+'/Output.csv')[data_obj.OutputVars]
                    i_df   = pd.DataFrame( np.arange(x_df.shape[0]), columns=['indx'])

                    # if (data_id != 'res'):
                    #     y    = y_df.to_numpy()
                    #     y    = (y - data_obj.system.C) / data_obj.system.D
                    #     y_df = pd.DataFrame(y, columns=y_df.columns)

                    data[data_type][data_id] = []
                    data[data_type][data_id].append(x_df) #.to_numpy()
                    data[data_type][data_id].append(y_df) #.to_numpy()
                    data[data_type][data_id].append(i_df) #.to_numpy()

            return data['train'], data['valid']

        #-----------------------------------------------------------------------



        #-----------------------------------------------------------------------
        def generate_training_data(data_obj, ic_list):
            print('\n[ROMNet]:   Generating training pts ...')

            data = {}
            for i, data_type in enumerate(['train', 'valid']):
                ic = ic_list[i]

                data_list = jl.Parallel(n_jobs=-1)( jl.delayed(single_train_scenario)(data_obj, i, ic_i) for ic_i in tqdm(ic) )
                #data_list = [ single_train_scenario(data_obj, i, ic_i) for ic_i in enumerate(tqdm(ic)) ]

                
                for j, data_j in enumerate(data_list):
                    data = stack_data(data, data_j, data_type)


                for data_id in data[data_type]:

                    if data[data_type][data_id] is None:
                        continue
                    x      = data[data_type][data_id][0]
                    x_df   = pd.DataFrame( x, columns=data_obj.system.names)

                    if (data_id != 'res'):
                        y      = data[data_type][data_id][1]
                        y_df   = pd.DataFrame( y, columns=data_obj.system.other_names)

                    else:
                        y_df   = pd.DataFrame( np.zeros((x.shape[0],len(data_obj.system.other_names))), columns=data_obj.system.other_names)
                        
                    i_df       = pd.DataFrame( np.arange(x_df.shape[0]), columns=['indx'])

                    data[data_type][data_id] = []
                    data[data_type][data_id].append(x_df) #.to_numpy()
                    data[data_type][data_id].append(y_df) #.to_numpy()
                    data[data_type][data_id].append(i_df) #.to_numpy()

                    path = data_obj.PathToDataFld+'/'+data_type+"/"+data_id+'/'
                    if not os.path.exists(path):
                        os.makedirs(path)
                    x_df.to_csv( path+"Input.csv",  index=False, float_format='%.12e' )
                    y_df.to_csv( path+"Output.csv", index=False, float_format='%.12e' )


            return data['train'], data['valid']

        #-----------------------------------------------------------------------



        #-----------------------------------------------------------------------
        def get_norm(data):

            FirstFlg = True
            for i in range(2):
                for data_id in data[i]:
                    if data[i][data_id] is None:
                        continue

                    x_df     = data[i][data_id][0]
                    xnorm    = x_df if FirstFlg else xnorm.append(x_df, ignore_index=True)

                    if (data_id != 'res'):
                        y_df     = data[i][data_id][1]
                        ynorm    = y_df if FirstFlg else ynorm.append(y_df, ignore_index=True)

                    FirstFlg = False

            return xnorm, ynorm
        #-----------------------------------------------------------------------



        #-----------------------------------------------------------------------
        def single_train_scenario(data_obj, i, ic_i):

            # Solving the PDE
            sol      = data_obj.system.solve(ic_i)
            t_i, y_i = sol[0], sol[1]

            # Define data type
            n_pts  = data_obj.n_train if i == 0 else data_obj.n_valid

            # Sample data
            data_i = get_data(ic_i, t_i, y_i, n_pts, data_obj.distribution)

            return data_i

        #-----------------------------------------------------------------------



        #-----------------------------------------------------------------------
        def get_data(ic, t, y, n_pts, distribution):

            tree = KDTree(np.c_[t.ravel()])
            data = {}
            for data_id, n in n_pts.items():
                if n == 0 or n is None:
                    data[data_id] = None
                else:
                    if data_id == 'ics':
                        idx = [0]
                    else:
                        idx = sampling(0, t.shape[0]-1, t, tree, n, distribution)
                    if data_id == 'res':
                        y = None
                    data[data_id] = extract_data_scenario(idx, ic, t, y)
                    
            return data

        #-----------------------------------------------------------------------



        #-----------------------------------------------------------------------
        def sampling(idx_min, idx_max, t, tree, n_pts, distribution):
            '''Get data by sampling t vector.'''
            
            t_min, t_max = t[idx_min], t[idx_max]
            if distribution == 'random':
                _, ii  = tree.query([[t_min],[t_max]], k=1)
                idx    = np.sort(np.random.choice(ii[1]-ii[0], n_pts, replace=False)) + ii[0]
            else:
                t_vec  = np.linspace(t_min, t_max, n_pts, dtype=np.float64)
                h      = (t_vec[1,0]-t_vec[0,0])*0.9
                dlt    = np.vstack(([[0.]], (np.random.rand(n_pts-2,1)-0.5)*h, [[0.]]))
                t_unif = t_vec + dlt
                _, idx = tree.query(t_unif, k=1)

            return idx

        #-----------------------------------------------------------------------



        #-----------------------------------------------------------------------
        def extract_data_scenario(idx, ic, t, y):

            m    = np.shape(idx)[0]
            u, t = np.tile(ic, (m,1)), t[idx]

            if y is None:
                return [np.concatenate([t, u], axis=1)]
            else:
                return [np.concatenate([t, u], axis=1), y[idx]]

        #-----------------------------------------------------------------------



        #-----------------------------------------------------------------------
        def stack_data(data, data_new, data_type):
            
            if data_type not in data:
                data[data_type] = data_new
            else:
                for data_id in data[data_type]:
                    if data[data_type][data_id] is None:
                        continue
                    for i, xyi in enumerate(data[data_type][data_id]):
                        xyi_new = data_new[data_id][i]
                        if isinstance(xyi, (list,tuple)):
                            xyi = list(map(lambda x,y,ii: np.vstack((x,y,ii)), xyi, xyi_new))
                        else:
                            xyi = np.vstack((xyi,xyi_new))
                        data[data_type][data_id][i] = xyi

            return data

        #-----------------------------------------------------------------------



        #-----------------------------------------------------------------------
        def unpack_data(data):
            _data = []
            for xyi in data:
                if isinstance(xyi, (list,tuple)):
                    _data.extend(xyi)
                else:
                    _data.append(xyi)
            return _data

        #-----------------------------------------------------------------------



        #-----------------------------------------------------------------------
        def read_testing_data(data_obj):
            print('\n[ROMNet]:   Reading test pts ...')


            # Print ics
            Data    = pd.read_csv(data_obj.PathToDataFld+'/test/ics.csv')
            ic_test = Data.to_numpy()
    
            data_test = [single_test_scenario(self, i, ic_i) for i, ic_i in enumerate(tqdm(ic_test))]
            
            return data_test

        #-----------------------------------------------------------------------



        #-----------------------------------------------------------------------
        def generate_testing_data(data_obj, ic_test, ic_train=None, header=None):
            print('\n[ROMNet]:   Generating test pts ...')

            # Test underfitting
            if (ic_train is not None):
                assert data_obj.n_scenarios >= data_obj.n_test
                _, idx  = data_obj.split_idxs(ic_train.shape[0], data_obj.n_test)
                ic_test = np.vstack((ic_test, ic_train[idx]))

            # Print ics
            Data = pd.DataFrame(ic_test, columns=data_obj.system.other_names)
           
            # Print input space
            path = data_obj.PathToDataFld+'/test/'
            if not os.path.exists(path):
                os.makedirs(path)
            Data.to_csv(path+"/ics.csv", index=False, float_format='%.10e' )

            # Generate data
            #data_test = jl.Parallel(n_jobs=-1)(jl.delayed(single_test_scenario)(self, i, ic_i) for i, ic_i in enumerate(tqdm(ic_test)) )
            data_test = [single_test_scenario(self, i, ic_i) for i, ic_i in enumerate(tqdm(ic_test))]
            
            return data_test

        #-----------------------------------------------------------------------



        #-----------------------------------------------------------------------
        def single_test_scenario(data_obj, i, ic_i):

            # Solving the ODE
            sol      = data_obj.system.solve(ic_i)
            t_i, y_i = sol[0], sol[1]

            # Extract data values
            tree     = KDTree(np.c_[t_i.ravel()])
            d, ii    = tree.query([[data_obj.system.t0],[data_obj.system.tEnd]], k=1)
            t_i, y_i = t_i[ii[0,0]:ii[1,0]], y_i[ii[0,0]:ii[1,0]]
            u_i      = np.tile(ic_i, (t_i.shape[0],1))

            # Collect data values
            data_i   = [np.concatenate([t_i, u_i], axis=1), y_i]

            return data_i

        #-----------------------------------------------------------------------


        self.PathToDataFld = InputData.PathToDataFld
        if not os.path.exists(self.PathToDataFld):
            os.makedirs(self.PathToDataFld)


        self.distribution   = InputData.DataDist    

        self.data_ids       = list(InputData.NTrain.keys())

        self.BatchSize      = InputData.BatchSize
        self.ValidBatchSize = InputData.ValidBatchSize

        # Data Quantities -----------------------------------------------------
        # Training/Validation pts
        self.n_train     = InputData.NTrain
        try:
            self.n_valid = InputData.NValid
        except:
            valid_perc   = np.power(InputData.ValidPerc/100., len(self.n_train))
            self.n_valid = { k: round(valid_perc*self.n_train[k]) for k in self.n_train}
            self.n_train = { k: self.n_train[k] - self.n_valid[k] for k in self.n_train}
        for k in self.n_train:
            if self.n_valid[k] == 0 and self.n_train[k] != 0:
                self.n_valid[k] = 1
        

        # Testing scenarios
        self.n_test            = InputData.NTest
       
        self.test_flg          = InputData.TestFlg   
    
        self.other_ranges      = self.system.other_ranges


        if (InputData.GenerateFlg):

            ic_list                = initial_cond(self)            
            train_data, valid_data = generate_training_data(self, ic_list)
            if (self.test_flg):
                test_data          = generate_testing_data(self,  ic_list[2])
            else:
                test_data          = None
        else:

            train_data, valid_data = read_training_data(self)
            if (self.test_flg):
                test_data          = read_testing_data(self)
            else:
                test_data          = None


        self.xnorm, self.ynorm = get_norm([train_data, valid_data])
        self.transform_normalization_data()
        self.compute_statistics()      


        self.n_train_tot = {}
        for key, value in train_data.items():
            self.n_train_tot[key] = len(value[0])


        # Training/Validation data
        if valid_data:
            # train_dset = self.data_as_dict(train_data)
            # valid_dset = self.data_as_dict(valid_data)
            train_dset = train_data
            valid_dset = valid_data
        elif validation_split:
            self.validation_split  = validation_split
            train_dset, valid_dset = self.get_train_valid( train_data)
        else:
            train_dset = train_data
            valid_dset = None


        print("[ROMNet]:   Train      Data: ", train_data)
        print("[ROMNet]:   Validation Data: ", valid_data)

        self.train     = DataSequence(train_dset, self.data_ids, batch_size=self.BatchSize)
        if valid_dset:
            self.valid = DataSequence( valid_dset, self.data_ids, batch_size=self.BatchSize if self.ValidBatchSize  is None else self.ValidBatchSize )

        # Testing data
        self.test = test_data

        # Update data ids
        self.data_ids = self.train.data_ids
        if self.valid:
            self.data_ids_valid = self.valid.data_ids


        print("[ROMNet]:   Train      Data: ", self.train)
        print("[ROMNet]:   Validation Data: ", self.valid)

    #===========================================================================



    #===========================================================================
    def get_num_pts(self, data_type='training', verbose=1):

        def print_fn(dset, data_type, verbose):
            num_pts = {data_id: dset.n_samples[data_id] for data_id in dset.data}
            if verbose:
                print("Number of pts for data " + k + ": ", v)
                for k, v in num_pts.items():
                    utils.print_submain("  - '%s': %8d" % (k, v))
            return num_pts

        super(PDE, self).get_num_pts( data_type=data_type, verbose=verbose, print_fn=print_fn )

    #===========================================================================



    #===========================================================================
    def get_train_valid(self, data):

        train, valid = {}, {}
        for i, data_i in data:
            train[i], valid[i] = super(PDE, self).get_train_valid(data_i)

        return train, valid

    #===========================================================================



    #===========================================================================
    def res_fn(self, net):
        '''Residual loss function'''

        self.NVarsx      = net.NVarsx
        self.NVarsy      = net.NVarsy

        def residual(inputs, training=True):


            other_vars, *ind_vars = tf.split(inputs, self.size_splits, axis=1)

            # Evaluate gradients
            grads = self.grad_fn(self, self.order, net, other_vars, ind_vars, training)
            # if (self.NVarsy > 1):
            #     grads = [tf.split(g, self.NVarsy, axis=1) for g in grads]

            # Evaluate residual
            return self.get_residual(other_vars, ind_vars, grads)
        
        return residual

    #===========================================================================
