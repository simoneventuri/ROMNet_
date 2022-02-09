# Neural-Network-Based Surrogates for Reduced Order Model Dynamics (ROMNet)



--------------------------------------------------------------------------------------
## Executing RomNET:

1. In $WORKSPACE_PATH, create a ROMNET folder

2. Inside $WORKSPACE_PATH/ROMNet/, clone the ROMNet repository and rename it "romnet"

	Note: $WORKSPACE_PATH
					├── ...
					├── ROMNET
					│		└── romnet
									├── app
									├── database
									├── ...

3. From $WORKSPACE_PATH/ROMNet/romnet/, install the code (i.e., $ python3 setup.py install)

4. From $WORKSPACE_PATH/ROMNet/romnet/app/, launch the code (i.e., $ python3 RomNet.py path-to-input-folder) 
	(e.g. fpython3 RomNet.py ../input/MassSpringDamper/DeepONet/)





--------------------------------------------------------------------------------------
## Test Cases:

- Mass-Spring-Damper System (MassSpringDamper)
	
	----------------------------------------------------------------------------------
	1. Data-driven deep operator network (DeepONet) for predicting position and velocity 
		1.1. Generate data by running $WORKSPACE_PATH/ROMNet/romnet/scripts/generating_data/MassSpringDamper/Generate_Data_1.ipynb
		1.2. In $WORKSPACE_PATH/ROMNet/romnet/input/MassSpringDamper/DeepONet/ROMNet_Input.py, change:
			1.2.1. "self.WORKSPACE_PATH = ..." 
		1.3. Move to $WORKSPACE_PATH/ROMNet/romnet/app/ and run "python3 ROMNet.py ../input/MassSpringDamper/DeepONet/"
		1.4. Postprocess results via $WORKSPACE_PATH/ROMNet/romnet/scripts/postprocessing/MassSpringDamper/DeepONet/Predict_DeepONet.ipynb



	----------------------------------------------------------------------------------
	2. Physics-informed deep operator network (DeepONet) for predicting position and velocity 
		2.1. Generate data by running $WORKSPACE_PATH/ROMNet/romnet/scripts/generating_data/MassSpringDamper/Generate_Data_1.ipynb
		2.2. In $WORKSPACE_PATH/ROMNet/romnet/input/MassSpringDamper/DeepONet/ROMNet_Input.py, change:
			2.2.1. "self.WORKSPACE_PATH = ..." 
			2.2.2. "self.n_train      = {'ics': 0, 'res': 0"
			2.2.3. "self.losses      = {'ics': {'name': 'MSE', 'axis': 0}, 'res': {'name': 'MSE', 'axis': 0}}" 
			2.2.4. "self.loss_weights = {'ics': 1., 'res': 1.}" 
		2.3. Move to $WORKSPACE_PATH/ROMNet/romnet/app/ and run "python3 ROMNet.py ../input/MassSpringDamper/DeepONet/"
		2.4. Postprocess results via $WORKSPACE_PATH/ROMNet/romnet/scripts/postprocessing/MassSpringDamper/DeepONet/Predict_DeepONet.ipynb



	----------------------------------------------------------------------------------
	3. POD-based interpreation of DeepONets:
		3.1. Generate     data by running $WORKSPACE_PATH/ROMNet/romnet/scripts/generating_data/MassSpringDamper/Generate_Data_1.ipynb
		3.2. Generate POD data by running $WORKSPACE_PATH/ROMNet/romnet/scripts/generating_data/MassSpringDamper/Generate_Data_2_All.ipynb
		3.3. Train Trunk:
			3.3.1. In $WORKSPACE_PATH/ROMNet/romnet/input/POD/MassSpringDamper/FNN/Trunk/ROMNet_Input.py, change:
				3.3.1.1. "self.WORKSPACE_PATH = ..." 
			3.3.2. Move to $WORKSPACE_PATH/ROMNet/romnet/app/ and run "python3 ROMNet.py ../input/POD/MassSpringDamper/FNN/Trunk/"
			3.3.3. Postprocess results via $WORKSPACE_PATH/ROMNet/romnet/scripts/postprocessing/POD/MassSpringDamper/FNN/Predict_FNN_Trunk.ipynb
		3.4. Train Branch  1:
			3.4.1. In $WORKSPACE_PATH/ROMNet/romnet/input/POD/MassSpringDamper/FNN/Branch/ROMNet_Input.py, change:
				3.4.1.1. "self.WORKSPACE_PATH = ..." 
				3.4.1.2. "self.i_red = 1"
			3.4.2. Move to $WORKSPACE_PATH/ROMNet/romnet/app/ and run "python3 ROMNet.py ../input/POD/MassSpringDamper/FNN/Branch/"
			3.4.3. Change "i_red = 1" in $WORKSPACE_PATH/ROMNet/romnet/scripts/postprocessing/POD/MassSpringDamper/FNN/Predict_FNN_Branch.ipynb
			3.4.4. Postprocess results via $WORKSPACE_PATH/ROMNet/romnet/scripts/postprocessing/POD/MassSpringDamper/FNN/Predict_FNN_Branch.ipynb
		3.5. Train Branch  2:
			3.5.1. In $WORKSPACE_PATH/ROMNet/romnet/input/POD/MassSpringDamper/FNN/Branch/ROMNet_Input.py, change:
				3.5.1.1. "self.WORKSPACE_PATH = ..." 
				3.5.1.2. "self.i_red = 2"
			3.5.2. Move to $WORKSPACE_PATH/ROMNet/romnet/app/ and run "python3 ROMNet.py ../input/POD/MassSpringDamper/FNN/Branch/"
			3.5.3. Change "i_red = 2" in $WORKSPACE_PATH/ROMNet/romnet/scripts/postprocessing/POD/MassSpringDamper/FNN/Predict_FNN_Branch.ipynb
			3.5.4. Postprocess results via $WORKSPACE_PATH/ROMNet/romnet/scripts/postprocessing/POD/MassSpringDamper/FNN/Predict_FNN_Branch.ipynb
		Note: If correctly executed, Predict_FNN_Branch.ipynb and Predict_FNN_Trunk.ipynb created the file: $WORKSPACE_PATH/ROMNet/Data/MSD_100Cases/Orig/All/FNN/Final.h5, 
			  which contains the trained parameter values for branches and trunk.
		3.6. Transfer to DeepONet:
			3.6.1. In $WORKSPACE_PATH/ROMNet/romnet/input/MassSpringDamper/DeepONet/ROMNet_Input.py, change:
				3.6.1.1. "self.WORKSPACE_PATH = ..." 
				3.6.1.2. "self.path_to_load_fld = ... " by pointing it to $WORKSPACE_PATH/ROMNet/Data/MSD_100Cases/Orig/All/FNN/Final.h5
				3.6.1.3. "self.trainable_flg = {'DeepONet': 'none'}"
			3.6.3. Move to $WORKSPACE_PATH/ROMNet/romnet/app/ and run "python3 ROMNet.py ../input/MassSpringDamper/DeepONet/"
			3.6.4. Postprocess results via $WORKSPACE_PATH/ROMNet/romnet/scripts/postprocessing/MassSpringDamper/DeepONet/Predict_DeepONet.ipynb




--------------------------------------------------------------------------------------
## Implemented NN Algorithms:

- Deterministic Neural Networks

- Probabilistic Neural Networks

	- Monte Carlo Dropout

	- Bayes by Backprop