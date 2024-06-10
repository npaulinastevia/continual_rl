# Continuously Learning Bug Locations
###Requirements:
Install Python (version >= 3).
Install all the packages listed in the requirements.txt file.
###Datasets

As discuss in the paper, we provide the dataset to train the CL agents on the non-stationary task. For example the dataset for Aspect is called AspectJ_test_NS.csv (test file) and AspectJ_train_NS.csv (train file). 
As for the other projects we provide the whole data to be split (60:40) according to the algorithm on the paper.


###Bug inducing factors

We provide the regression model used to enhanced the capability of the CL agents, in the zip file regression_model

###Baseline studies

FLIM https://github.com/hongliangliang/flim

RLOCATOR https://zenodo.org/records/11265302
To obtain the stationary data, refer to RLOCATOR implementation

###Training and testing of the CL agents

For example for Tomcat project:

run ewc.sh for EWC CL agent

run clear.sh for CLEAR CL agent

###Training and testing of the CL agents with logistic regression

For example for Tomcat project:

run enhanced_ewc.sh for EWC CL agent

run enhanced_clear.sh for CLEAR CL agent


