# UNet-with-RNN

This repo implements U-Net models combined with various Recurrent Neural Networks (LSTM and GRU). In other words the model takes a 3D data as input and trains a U-Net combined with RNN (U-Net RNN) that learns to predict the next frame in a sequence. The U-Net RNN model is basically a standard U-Net model for each time sequence, except some of the convolutions are changed to convolutional RNNs. An example case can be seen in the below figure. 

![image](https://github.com/cemdaloglu/UNet-with-RNN/assets/36455629/c9814c07-75b2-476c-88d9-60265f93f7e4)

In this repo, there are 7 experiments that are already implemented.

- Baseline: Takes 3 previous time steps data and their predictions as input to a standard U-Net model and predicts the next frame.
1. Same architecture as baseline, except that each resolution stage in the encoder consists of 1 conv-LSTM after the highest resolution stage.
2. Same as 1, but with 2 conv-LSTM.
3. Same as 1, but with 1 conv layer and 1 conv-LSTMs.
4. Same as 1, but all conv-LSTMs are replaced by peephole conv-LSTMs.
5. Same as 1, except that the two highest resolution stages stay unchanged.
6. Same as 2, but all conv-LSTMs are replaced by conv-GRUs

# Data

- Dataset is not provided.
- DataClass function in helper_funcs.py file is hard coded for input and label/target data. It should be changed according to your data.
- input shape is (time sequences, channels, depth, height, width), and the target/label shape is (time sequences, depth, height, width).

# Dependencies

A lot.
