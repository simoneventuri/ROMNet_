import sys
print(sys.version)
import os
import numpy as np
import pandas as pd
import time
import shutil

WORKSPACE_PATH = os.environ['WORKSPACE_PATH']
# WORKSPACE_PATH = os.getcwd()+'/../../../../../'

ROMNet_fld     = WORKSPACE_PATH + '/ROMNet/romnet/'

from joblib import Parallel, delayed
import multiprocessing
import itertools



NVars            = 2
NProcs           = 2

DRAlog           = 'PCANorm'
NRODs            = 2

path_to_orig_fld_branch = ROMNet_fld + '/input/ScenarioAggregated_ROMs/MassSpringDamper/FNN/Branch/'
path_to_orig_fld_trunk  = ROMNet_fld + '/input/ScenarioAggregated_ROMs/MassSpringDamper/FNN/Trunk/'
path_to_run_fld         = ROMNet_fld + '/../MSD_100Cases_'+str(NRODs)+DRAlog+'/'  
  


def copy_params_branch(ROMNet_fld, path_to_run_fld, iVar):

	os.system('python3 ./Merge_BranchHDF5s_.py '+str(iVar+1))



def copy_params_trunk(ROMNet_fld, path_to_run_fld, iVar):

	os.system('python3 ./Merge_TrunkHDF5s_.py '+str(iVar+1))




shutil.copyfile(path_to_orig_fld_branch+'/Merge_BranchHDF5s.py', './Merge_BranchHDF5s_.py')

results = Parallel(n_jobs=1)(delayed(copy_params_branch)(ROMNet_fld, path_to_run_fld, iVar) for iVar in range(NVars))



shutil.copyfile(path_to_orig_fld_trunk+'/Merge_TrunkHDF5s.py', './Merge_TrunkHDF5s_.py')

results = Parallel(n_jobs=1)(delayed(copy_params_trunk)(ROMNet_fld, path_to_run_fld, iVar) for iVar in range(NVars))