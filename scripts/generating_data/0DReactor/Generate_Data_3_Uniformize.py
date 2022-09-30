### Importing Libraries

import sys
print(sys.version)
import os
import time


### Defining WORKSPACE_PATH
WORKSPACE_PATH = os.environ['WORKSPACE_PATH']
#WORKSPACE_PATH = os.path.join(os.getcwd(), '../../../../../../')
ROMNet_fld     = os.path.join(WORKSPACE_PATH, 'ROMNet/romnet/')


### Importing External Libraries
import numpy                             as np
import pandas                            as pd
#from alive_progress                  import alive_bar



DataDir      = os.path.join(WORKSPACE_PATH, 'ROMNet/Data/0DReact_Isobaric_1000Cases_H2/')
NIC          = 1
QoIName      = 'N'
NPoints      = 1000


#with alive_bar(NIC) as bar:
for iIC in range(NIC):

	Data         = pd.read_csv(DataDir+'/Orig/train/ext/y.csv.'+str(iIC+1))
	Data         = Data.clip(lower=1.e-30, upper=1.e10)
	#SpeciesNames = list(pd.read_csv(DataDir+'/Orig/train/ext/CleanVars.csv').columns)
	SpeciesNames = list(Data.columns)[1::]

	YY           = Data[SpeciesNames].to_numpy()



	t        = np.log10(Data['t'].to_numpy())
	QoI      = np.log10(np.clip(Data[QoIName].to_numpy(), 1.e-30, 1.e5))
	yGrid    = np.flip( np.linspace(np.log10(Data[QoIName].min()), np.log10(Data[QoIName].max()), NPoints) )



	iChange  = [False]
	for it in range(1,len(QoI)-1):
	    iChange.append( not ((QoI[it] - QoI[it-1])*(QoI[it+1] - QoI[it]) > 0) )
	iChange.append(False)

	Locs     = np.append(np.append([0], np.where(iChange)), [len(QoI)-2])



	QoIRange     = np.abs(QoI.max()-QoI.min())

	xGrid        = []
	yGrid_       = []
	for iLoc in range(len(Locs)-1):
	    t_        = t[Locs[iLoc]:Locs[iLoc+1]+1]
	    QoI_      = QoI[Locs[iLoc]:Locs[iLoc+1]+1]
	    QoIRange_ = np.abs(QoI[Locs[iLoc]] - QoI[Locs[iLoc+1]+1])
	    NPoints_  = np.round(QoIRange_/QoIRange*NPoints)
	    if (NPoints_ > 1):  
	        if (QoI[Locs[iLoc]] > QoI[Locs[iLoc+1]+1]):
	            Sign   =  1.
	        else:
	            Sign   = -1.                    
	        hy         = -(QoI[Locs[iLoc]] - QoI[Locs[iLoc+1]+1]) / (NPoints_-1)
	        yGrid      = np.arange(QoI[Locs[iLoc]], QoI[Locs[iLoc+1]+1], hy)  
	        iOrig      = Locs[iLoc]
	        for y in yGrid:
	            while ((QoI[iOrig] - y) * Sign >= 0) and (iOrig <= len(QoI)-2):
	                iOrig += 1
	            iOrig -= 1
	            Deltax = t[iOrig+1]  - t[iOrig]
	            Deltay = (QoI[iOrig+1] - QoI[iOrig])
	            yT     = (y - QoI[iOrig])
	            xT     = yT / Deltay * Deltax
	            x      = t[iOrig] + xT
	            xGrid.append(x)
	            yGrid_.append(y)
	            
	    else:
	        xGrid.append(t[Locs[iLoc]])
	        yGrid_.append(QoI[Locs[iLoc]])



	YY_   = np.zeros((len(xGrid), len(SpeciesNames)))
	iOrig = 0 
	for ix, x in enumerate(xGrid):
	    while (x >= t[iOrig]) and (iOrig <= len(t)-2):
	        iOrig += 1
	    iOrig -= 1
	    for iSpec in range(YY.shape[1]):
	        Deltax        = t[iOrig+1] - t[iOrig]
	        Deltay        = np.log10(YY[iOrig+1,iSpec]) - np.log10(YY[iOrig,iSpec])
	        xT            = x - t[iOrig]
	        YY_[ix,iSpec] = 10.**( np.log10(YY[iOrig,iSpec]) + xT * Deltay / Deltax )
	        
	DataNew_      = pd.DataFrame(YY_, columns=SpeciesNames)
	DataNew_['t'] = 10.**np.array(xGrid)

	if (iIC == 0):
		DataNew = DataNew_.copy()
	else:
		DataNew = pd.concat([DataNew, DataNew_], axis=0)



FileName = DataDir+'/Orig/train/ext/Uniformized.csv'
DataNew[SpeciesNames].to_csv(FileName, index=False)