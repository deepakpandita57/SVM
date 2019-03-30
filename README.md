# Implement SGD for SVM for the adult income dataset. Experiment with performance as a function of the capacity parameter C.


Task
=============================================================================================================
Predict whether income exceeds $50K/yr based on census data. Also known as "Census Income" dataset.

First column is class label, remaining columns are a sparse representation
of the feature vector in format <feature>:<value>.  All other features are 0.

more information on the task:
http://archive.ics.uci.edu/ml/datasets/Adult

preprocessed version from: 
http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html


Files
=============================================================================================================
"SVM.py" contains the code for SVM
"plot.py" contains the code to plot the accuracies on test and dev data while varying Capacity
"plot.png" contains the plot. There are 40 points in the plot from 10^-3 to 10^4, each c is 1.5 times of the previous one
"README"


Algorithm
=============================================================================================================
Support Vector Machine with Stochastic Gradient Descent is implemented.


Instructions for running "SVM.py"
=============================================================================================================
To run the script "SVM.py" type "python3 SVM.py" in the commandline
The default number of epochs is 1 and capacity is 0.868.

We can also specify the number of epochs to run using the optional argument "--epochs" and
capacity using the optional argument "--capacity"


Results
=============================================================================================================
After 10 epochs and 0.868 capacity the SVM gives a test accuracy of 0.831 which is better than what we got using perceptron.
If we use just 1 epoch and 0.868 capacity we are getting a test accuracy of 0.833
With 10 epochs and 0.1 capacity we are getting a test accuracy of 0.845


Interpretation
=============================================================================================================
While keeping epochs constant at 5 and increasing the capacity from 10^-3 to 10^4, the accuracy on test set first increases in the range (10^-3 to <10^-1).
After 10^-1 it starts dropping and then it starts fluctuating(near 10^1). This shows that the capacity is important and 
it provides the best performance in the range 10^-2 to 10^-1 on this dataset.  
