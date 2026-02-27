# Lab 3

## docs
Documentation containing external materials. For the curious ones.   

## project
Code, everything written in python. Includes Makefile.  

### installation & execution
Navigate to the same directory as makefile (./project).  

"make setup". Installs all python dependencies in a python virtual environment (5GB size).  
"make run". Run project with python virtual environment (check virtual env. and dependencies need to install).  
"make clean". Deletes all downloaded dependencies (virtual environment and pycache). When done delete the datasets manually by refering to the path in your terminal!  

#### with CUDA.
Installation of tensorflow will be with CUDA support. However installing CUDA package for GPU will not be included in makefile because of hardware limitations.  
https://www.tensorflow.org/install/pip.  

If CUDA tools not installed on machine, in ./project/python_requirements.txt change tensorflow[and-cuda] to only tensorflow.  
Some bugs may occur with tensorflow[and-cuda] when ran without installed cuda packages on local machine.  

#### Dataset deletion.
Currently inorder to delete downloaded datasets, refer to path /home/{USERNAME}/.cache/kagglehub/datasets.  
Exact path to dataset folder is printed in terminal during project execution and dataset configuration.  