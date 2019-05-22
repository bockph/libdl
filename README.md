The XOR problem is simulated in the build File Example_XOR, which is build from the Code in Example_XOR.cpp
When executing, it uses the following Mini-batch where one data point is represented by one row:
Eigen::MatrixXf mX1(4, 2);
	mX1 << 1, 0,
			0, 1,
			0, 0,
			1, 1;
with the Corresponding Classes stored in the Vector C
Eigen::MatrixXf C(4, 1);
	C << 1,
			1,
			0,
			0;

and randomly initialized Weights and Biases.

As a output it gives first the results of the first forward pass and the loss of the first forward pass.
Second, it gives the result and Loss after 5000x forward pass & backprop