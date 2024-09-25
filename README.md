# ipdps-25

## For The CWRU Pioneer Cluster

Make sure that your environment is setup as follows:

- module load Python/3.10.8-GCCcore-12.2.0
- Utilize PyTorch 2.4.0 compiled with CUDA 12.1 in your virtual environment
- module load CUDA/12.1.1
- module load Ninja/1.11.1-GCCcore-12.2.0

I will use the /Sparse_FlashAttention_CSR/ folder as an example for creating a function bound to PyTorch.

This folder should be copied under the torch folder in your virtual environment. My path for example:

- /home/nkt8/cust_pt/testing/lib/python3.10/site-packages/torch/

The *.cpp file is in charge of actually creating the binding and performing PyTorch's checks on the input data. The *.cu file contains the kernel and the function that calls the kernel. setup.py is what you actually run to compile the function.

You'll notice that setup.py points to the *.cpp and *.cu file and gives the module a name, in my case I called it **spfa_csr**. This will be the name of the package itself.

Please use mine as a template, but for further info you can refer to: https://pytorch.org/tutorials/advanced/cpp_extension.html

Once you have compiled your example, please use the testing script located in /verification/ to verify that your output matches what PyTorch outputs. You will need to import your own function and call it as well to verify. Please make sure that you are using the MATH backend (what should be uncommented within the code).

Please note that the verification process is done with fully dense matrices to verify that our process exactly matches that of attention.