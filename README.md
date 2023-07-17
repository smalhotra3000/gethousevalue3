# Housing_value

This is a python library created by Sameer Malhotra as part of the MLE training in TigerAnalytics.

## Installation

To install this library using the sdist (tar.gz), follow the following steps:
1. Download the latest version of the *gethousevalue3.tar.gz* file and extrat the contents
2. Open the terminal and navigate to the directory where the above files are extracted
3. Install the library using the pip command: "pip install ."
   (Make sure you are in the root directory containing the setup.py file)
4. Import the library in your python code to use it: "import gethousevalue"

To install this library using the .whl file, follow the following steps:
1. Open the terminal and navigate to the directory containing the .whl file
2. Run the command "pip install gethousevalue3-1.0.0-py3-none-any.whl"

To get the prequirequiste environment up and running, use the env.yml file and follow the following instructions:
1. Open the terminal and navigate to the directory containing the yml file
2. Run the command "pip env create --name myenv --file env.yml" to create the environment. This will install the dependencies.
3. Run the command "pip activate myenv" to activate the environment

# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data.

The following techniques have been used:

 - Linear regression
 - Decision Tree
 - Random Forest
