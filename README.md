# UNet-with-RNN

This code implements U-Net models combined with various Recurrent Neural Networks (RNN, LSTM, and GRU). In other words the model takes one text file as input and trains a Recurrent Neural Network that learns to predict the next character in a sequence. The RNN can then be used to generate text character by character that will look like the original training data.

Baseline: Model 3.
Same architecture as baseline, except that each resolution stage in the encoder consists of 1 conv-LSTM after the highest resolution stage.
Same as 1, but with 2 conv-LSTM.
Same as 1, but with 1 conv layer and 1 conv-LSTMs.
Same as 1, but all conv-LSTMs are replaced by peephole conv-LSTMs.
Same as 1, except that the two highest resolution stages stay unchanged.
Same as 2, but all conv-LSTMs are replaced by conv-GRUs

![image](https://github.com/cemdaloglu/UNet-with-RNN/assets/36455629/c9814c07-75b2-476c-88d9-60265f93f7e4)

