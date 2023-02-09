# KerasPreprocessings
KerasPreprocessings is a part of the DeepHLS toolchain.

# Citing us
If you find this work helpful for your research, please consider citing our paper published in The 27th IEEE International Conference on Electronics, Circuits and Systems (ICECS), 2020.

```
Riazati, Mohammad, Masoud Daneshtalab, Mikael Sjödin, and Björn Lisper. "DeepHLS: A complete toolchain for automatic synthesis of deep neural networks to FPGA." In The 27th IEEE International Conference on Electronics, Circuits and Systems (ICECS), pp. 1-4. IEEE, 2020.
```

# Description
For more information about the toolchain and how to use KerasPreprocessings, please refer to DeepHLS tool opensourced under the same github user page.

# Required packages
* numpy
* tensorflow
* keras
* tensorflow_datasets
* pydot

# How to use
* Clone the repository. 
* Create the "outputs" directory just beside KerasPreprocessings.py
* Determine the network and the dataset in the beginnig of the code
* Run
* Use output_arch.py as the input for DeepHLS to create the synthesizable C implementation
* Copy data.h and param.h to DeepSimulator folder as the input dataset and parameters.
* (Copy the file created by DeepHLS to DeepSimulator folder as the source C code for simulation)


