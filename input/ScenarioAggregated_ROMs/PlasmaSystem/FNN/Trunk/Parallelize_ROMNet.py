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



NVars            = 5
NProcs           = 5

DRAlog           = 'KPCA'
NRODs            = 32

path_to_orig_fld = ROMNet_fld + '/input/ScenarioAggregated_ROMs/PlasmaSystem/FNN/Trunk/'
path_to_run_fld  = ROMNet_fld + '/../PlasmaSyst_500Cases_'+str(NRODs)+DRAlog+'/'  
  


def run_variable(ROMNet_fld, path_to_run_fld, iVar):

	path_to_run_fld_ = path_to_run_fld + '/Var'+str(iVar+1)+'_Trunk/' 
	try:
	    os.makedirs(path_to_run_fld_)
	except:
	    pass           

	os.chdir(path_to_run_fld_)

	np.savetxt('./iVar.csv', np.array([[iVar+1]], dtype=int), fmt='%i')

	shutil.copyfile(ROMNet_fld+'/app/ROMNet.py', './ROMNet.py')
	shutil.copyfile(path_to_orig_fld+'/ROMNet_Input.py', './ROMNet_Input.py')

#	os.system('python3 ./ROMNet.py ./ > ./Out.txt')
	os.system('python3 ./ROMNet.py ./')



def copy_params(ROMNet_fld, path_to_run_fld, iVar):

	os.system('python3 ./Merge_TrunkHDF5s.py '+str(iVar+1))



try:
    os.makedirs(path_to_run_fld)
except:
    pass  



results = Parallel(n_jobs=NProcs)(delayed(run_variable)(ROMNet_fld, path_to_run_fld, iVar) for iVar in range(NVars))

os.chdir(path_to_run_fld)

shutil.copyfile(path_to_orig_fld+'/Merge_TrunkHDF5s.py', './Merge_TrunkHDF5s.py')

results = Parallel(n_jobs=1)(delayed(copy_params)(ROMNet_fld, path_to_run_fld, iVar) for iVar in range(NVars))
