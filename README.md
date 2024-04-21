# UNet-with-RNN

This repo implements U-Net models combined with various Recurrent Neural Networks (LSTM and GRU). In other words the model takes a 3D data as input and trains a U-Net combined with RNN (U-Net RNN) that learns to predict the next frame in a sequence. It is implemented for my master's thesis about Real-Time 3D Reconstruction of guidewires. The U-Net RNN model is basically a standard U-Net model for each time sequence, except some of the convolutions are changed to convolutional RNNs. An example case can be seen below.

# Real-Time 3D Reconstruction for Minimally Invasive Vascular Procedures

In minimally invasive vascular procedures, accurate image guidance is crucial for successful outcomes. Traditional 2D fluoroscopy lacks depth information, prompting the need for 3D real-time image guidance. However, achieving high temporal resolution with low X-ray dose presents a challenge.

This project addresses this challenge by proposing two novel approaches to improve the accuracy of 3D reconstruction algorithms for interventional materials. The first approach involves incorporating additional temporal information, such as 3D reconstructions from previous time steps, into the reconstruction pipeline. The second approach utilizes long-short-term memory (LSTM) blocks to enhance temporal consistency. The RNN model architecture can be seen in the below figure.

![image](https://github.com/cemdaloglu/UNet-with-RNN/assets/36455629/4487cf39-896e-41cb-bb74-e6a403f42ed5)


## Experiment Architectures

- **DTR_baseline:** Trained on a dataset comprising 16,000 training samples and 6,000 validation samples. It was tested on a separate set of 24,000 backprojection pairs to predict the corresponding 3D reconstructions, unseen during the training phase. The conducted three new experiments are the following.
- **DTR+PrevRecons:** This model was trained using the 16,000 backprojection pairs and their respective reconstructions from the DTR_baseline model's test set. To assess the performance of the DTR+PrevRecons model, we employed a set of 24,000 backprojection pairs for testing. Notably, the 16,000 prediction scenes used in this experiment were identical to those in the DTR baseline model's training set.
- **DTR+PrevOwnRecons:** For training, this model utilized the 16,000 backprojection pairs and their corresponding reconstructions obtained from the DTR+PrevRecons model's test set.
- **DTR+PrevGTs:** This model was trained using the 16,000 backprojection pairs alongside their respective ground truth reconstructions.

1. The architecture remains consistent with the `DTR_baseline`, with the exception that each resolution stage in the encoder incorporates one convLSTM except the highest resolution stage.
2. Similar to experiment 1, but with the inclusion of two convLSTMs at each resolution stage in the encoder.
3. In alignment with experiment 1, a comparable structure is maintained; however, it includes one convolution layer and one convLSTM at each resolution stage in the encoder.
4. Experiment 1's architecture is retained, but all convLSTMs are substituted with peephole convLSTMs.
5. Similar to experiment 1, but the two highest resolution stages remain the same.
6. Experiment 2's structure is mirrored, but all convLSTMs are substituted with convGRUs.
7. Experiment 3's configuration is maintained, with the inclusion of eight time steps as input.
8. Experiment 3's setup is replicated, but the loss function is evaluated only for the last time step.
9. Experiment 3's architecture is emulated, with the training epochs doubled.

Experimental results on a simulated guidewire dataset demonstrate significant improvements in reconstruction accuracy. Compared to the baseline algorithm, the proposed approaches achieve an enhanced Dice coefficient (DSC), with the second approach achieving the highest improvement to 79.04%. Tabular and graph of the results can be seen below.

![image](https://github.com/cemdaloglu/UNet-with-RNN/assets/36455629/45d202a4-ec59-4956-af8b-1ba036a94329)

All RNN experiments,

![image](https://github.com/cemdaloglu/UNet-with-RNN/assets/36455629/24e6ef1f-61a7-41b7-b418-e9ab3e3f318f)

Top-performing experiments,

![image](https://github.com/cemdaloglu/UNet-with-RNN/assets/36455629/5bec10ba-3506-4055-a2bf-22f20a2f9946)



Predictions of different models can be seen below,

![Untitled video - Made with Clipchamp](https://github.com/cemdaloglu/UNet-with-RNN/assets/36455629/37f9015d-c969-4944-8c7d-c3a7b1765d86)



# Data

- Dataset is not provided.
- DataClass function in helper_funcs.py file is hard coded for input and label/target data. It should be changed according to your data.
- input shape is (time sequences, channels, depth, height, width), and the target/label shape is (time sequences, depth, height, width).
- In the helper_funcs.py main function, first element of filterList parameter must be the channel size.

# Dependencies

A lot.
