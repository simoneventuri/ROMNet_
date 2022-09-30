# Instructions:

1. Generate Data in the Thermodynamic State Space (e.g.: via Generate_Data_1_Isobaric.py) 

2. Find the Main Modes of the Scenario-Aggregated State-Specific Evolutions (e.g.: via Generate_Data_5_NonLinear.py)

3. Fit the Modes and Generate N FNNs (i.e.: execute ./Parallelize_ROMNET.py)

4. Postprocess the FNNs and Check the Regression Errors (i.e., run /ROMNet/romnet/scripts/postprocessing/ScenarioAggregated_ROMS/0DReactor/FNN/Predict_FNN_Trunk.ipynb)

5. Transfer the FNNs' Parameters to Trunks' Parameters (i.e.: execute ./Merge_TrunkHDF5s.py)

6. Train DeepONet (e.g.: python3 ROMNet.py ../input/0DReactor/DeepONet/0DReactor_H2_TestCase12)
	Note: Change ../input/0DReactor/DeepONet/0DReactor_H2_TestCase12/ROMNet_Input.py and specify path to the just-generated hdf5 containing trunks' parameters
	
7. Postprocess the DeepONet (i.e.: Predict_DeepONet_Orig.ipynb)