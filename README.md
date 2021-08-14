# Neural-Network-Based Surrogates for Reduced Order Model Dynamics (ROMNet)



##------------------------------------------------------------------------------------
## Executing RomNET:

0 - In the ~/.bashrc File, Export the Enviroment Variable $WORKSPACE_PATH
(e.g., WORKSPACE_PATH='/home/sventur/WORKSPACE/', with this repository cloned in /home/sventur/WORKSPACE/ROMNet/)

1 - Generate data using the jupyter notebooks / python scripts in /ROMNet/romnet/scripts/generating_data/ 
(e.g., /ROMNet/romnet/scripts/generating_data/MassSpringDamper/Generate_Data.ipynb)

2 - Go to $WORKSPACE_PATH/ROMNet/romnet/app/

3 - Launch the Program:
$ python3 RomNet.py path-to-input-folder
(e.g., python3 RomNet.py $WORKSPACE_PATH/ROMNet/romnet/input/MassSpringDamper/DeepONet/Deterministic/)

4 - Post-process the surrogate during/after training via the scripts in /ROMNet/romnet/scripts/postprocessing/ 
(e.g., /ROMNet/romnet/scripts/postprocessing/MassSpringDamper/DeepONet/Predict_DeepONet.ipynb)



##------------------------------------------------------------------------------------
## Test Cases:

List of Test Cases are currently available:

- 2D Sinusoidal Function (Sinusoidal) 

- 2D Flame Data (FlameData)

- Mass-Spring-Damper System (MassSpringDamper)

- Perfectly Stirred Reactor System (PSReactor) (Still under Development)

- Isocoric Adiabatic 0D Reactor System (0DReactor) (Still under Development)



##------------------------------------------------------------------------------------
## Implemented NN Algorithms:

- Deterministic Neural Networks

- Probabilistic Neural Networks

	- Monte Carlo Dropout

	- Bayes by Backprop

	- Hamiltonian Monte Carlo