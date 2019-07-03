# libdl
# XOR Problem
The XOR problem is simulated in the build File Example_XOR, which is build from the Code in Example_XOR.cpp\
When executing, it uses the following Mini-batch where one data point is represented by one row:\
\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Eigen::MatrixXf mX1(4, 2);\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;mX1 << 1, 0,\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0, 1,\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0, 0,\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1, 1;\
			\
with the Corresponding Classes stored in the Vector C\
\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Eigen::MatrixXf C(4, 1);\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;C << 1,\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1,\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0,\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0;\
\
and randomly initialized Weights and Biases.\
\
As a output it gives first the results of the first forward pass and the loss of the first forward pass.\
Second, it gives the result and Loss after 5000x forward pass & backprop.\

# MNIST
The MNIST problem is simulated in the build File Example_MNIST, which is build from the Code in Example_MNIST.cpp\
The settings are predefined so that pre-learned weights are read from the WeightDeposit and are then used to predict a subset of the MNIST testset.
At the beginning of the main method of Example_MNIST.cpp these settings can be changed (as explained there), in order to train ones own weights.
