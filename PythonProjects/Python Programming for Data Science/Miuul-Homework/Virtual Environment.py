# Task 1: Create a virtual environment with your own name, install Python 3 during creation.
   # conda create -n barbar python=3
# Task 2: Activate the environment you created.
   #conda activate barbar
# Task 3: List the installed packages.
   # conda list
# Task 4: Download the current version of Numpy and version 1.2.1 of Pandas into the Environment at the same time.
   # conda install numpy pandas =1.2.1
# Task 5: What is the version of Numpy downloaded?
   #1.21.5
# Mission 6: Upgrade Pandas. What is the new version?
   #1.2.1
# Task 7: Delete numpy from environment.
   # conda remove numpy
# Task 8: Download the current versions of Seaborn and matplotlib library at the same time.
   # conda install seaborn matplotlib
# Task 9: Export the libraries in the virtual environment with their version information and examine the yaml file.
   # conda env export > environment.yaml
# Task 10: Delete the environment you created. First, deactivate the environment.
   # conda env remove -n barbar