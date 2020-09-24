#WSLSE for RL Parameter Recovery

##Iyer, E. S., Kairiss, M. A., Liu, A., Otto, A. R., & Bagot, R. C. (2020). Probing relationships between reinforcement learning and simple behavioral strategies to understand probabilistic reward learning. Journal of Neuroscience Methods

**Installation**

In order to use this you must clone this GitHub Repository to your local computer. 
Ensure that all packages are installed, notably rpy2. 

**Usage**

Once the GitHub Repo is cloned. You can begin RL parameter recovery. First open WSLSE.py and initialize the observable varibles as being either 1Back or 2Back. Then specify the paradigm and number of trials.
After initializing model selection, scroll to the bottom of the file. and enter the filepath that contains your win stay/lose shift and subject ID information. This file must take the form of a CSV with the headers as subject, win_stay, lose_shift (for 1Back) or subject, win_stay, lose_shift, win_stay2, lose_shift2 (for 2Back).
Then running the python file will output a csv (titled outputfile.csv by default) with the recovered RL parameters for each subject. 

**Adapting this code to other tasks, trial numbers, or RL models**

In order to adapt this code to other tasks, trial numbers, or RL models, one must generate the win stay, lose shift, and covariance numpy arrays specific to the task, number of trials, and RL model in question. This can be done by simply iterating over every combination of RL parameter. A simple version of this is script is located in the sample_data_1Back.py (to generate the 1back numpy arrays) and in sample_data_2 Back.py (to generate both the 1back and 2back numpy arrays). This function is named create_numpy_arrays(). To adjust trial numbers, edit the number in generate_data_PRL to reflect trial number. To adjust either the task or trial number, you must change the generate_data functions to reflect either the task or the RL model in question. Once the numpy arrays are generated, in WSLSE.py edit the function named init_model_1Back (for 1Back) or init_model_2Back so that when you specify your paradigm, it loads the numpy arrays you just generated. Once this is done, you should be able to proceed as outlined under USAGE to recover RL parameters. 
