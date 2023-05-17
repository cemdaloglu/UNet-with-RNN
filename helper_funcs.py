from torch.utils.data import Dataset
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from tqdm import tqdm
from torchmetrics import Accuracy, MetricCollection, Metric
import torch.nn as nn
from lstm_unet import UNet, UNetRNN
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os
from readWrite import loadNpz, readArray, writeArray
from tensorboardX import SummaryWriter
import random



def dataAugmentation(input5D, target4D, augPercent):
        # tmp = torch.zeros_like(input5D)
        flipSide = np.random.randint(2, 4)
        if np.random.uniform() > augPercent:
            nRot = np.random.randint(0, 4)
            tmp = torch.flip(torch.rot90(input5D, nRot, [3, 4]), [flipSide])
            targetArray = torch.flip(torch.rot90(target4D, nRot, [2, 3]), [flipSide-1])
        else:
            tmp = input5D
            targetArray = target4D
        return tmp, targetArray


def dataAugmentationBase(input5D, target4D, pred4D, augPercent):
        # tmp = torch.zeros_like(input5D)
        flipSide = np.random.randint(2, 4)
        if np.random.uniform() > augPercent:
            nRot = np.random.randint(0, 4)
            tmp = torch.flip(torch.rot90(input5D, nRot, [3, 4]), [flipSide])
            targetArray = torch.flip(torch.rot90(target4D, nRot, [2, 3]), [flipSide-1])
            predArray = torch.flip(torch.rot90(pred4D, nRot, [2, 3]), [flipSide-1])
        else:
            tmp = input5D
            targetArray = target4D
            predArray = pred4D
        return tmp, targetArray, predArray


class DataClass(Dataset):

    def __init__(self, dataPath, phase, trainPercent, no_of_seq, augPercent, isBase):
        self.dataPath = dataPath
        self.phase = phase
        self.multipPercent = trainPercent
        self.no_of_seq = no_of_seq
        self.augPercent = augPercent
        self.isBase = isBase
        if self.phase == 'val' or self.phase == 'test':
            self.multipPercent = 1 - trainPercent
        if self.isBase:
            self.beginIndex = 20000
        else:
            self.beginIndex = 0
        if self.phase == 'val' or self.phase == 'test':
            self.beginIndex = int(len(os.listdir(self.dataPath)) / 2 * (1 - self.multipPercent))
        self.mean, self.std = avg_mean_std(self.dataPath, 250)

    def __len__(self):
        return int(len(os.listdir(self.dataPath)) / 2 * self.multipPercent)

    def __getitem__(self, index):
        self.input = loadNpz(os.path.join(self.dataPath, "ex" + f"_{self.beginIndex + index}_" + "BP_guidewire_64x64x64x2x16.npz"))
        self.target = loadNpz(os.path.join(self.dataPath, "ex" + f"_{self.beginIndex + index}_" + "voxSTL_guidewire_64x64x64x16.npz"))
        if self.isBase:
            pred = readArray(os.path.join("C:\src\Predict_Results_12", "predict" + f"_{self.beginIndex + index - 20000}_" + "voxSTL_guidewire_64x64x64x13.raw"))

        if self.phase == 'test':
            nt = 12
        else:
            Nt = self.input.shape[0]
            nt = np.random.randint(3, Nt-self.no_of_seq)
        
        inputArrayLoad = self.input[ nt:nt+self.no_of_seq+1, :, :, :, :].astype(np.float32)
        if self.isBase:
            targetArrayLoad = np.where(self.target[nt+self.no_of_seq, :, :, :] > 1e-5, 1, 0)
        else:
            targetArrayLoad = np.where(self.target[nt:nt+self.no_of_seq+1, :, :, :] > 1e-5, 1, 0)
        inputArrayLoad = ( inputArrayLoad - self.mean ) / self.std
        inputArray = torch.from_numpy(inputArrayLoad).type(torch.float32)
        tmp_targetArray = torch.from_numpy(targetArrayLoad).type(torch.float32)

        if self.isBase:
            predArrayLoad = np.where(pred[nt-self.no_of_seq:nt, :, :, :] > 1e-5, 1, 0)
            tmp_predArray = torch.from_numpy(predArrayLoad).type(torch.float32)
            if tmp_targetArray.dim() == 3:
                tmp_targetArray = tmp_targetArray.unsqueeze(0)

        if self.phase=='train' and self.augPercent > 0.0:
            if self.isBase:
                tmp, targetArray, predArray = dataAugmentationBase(inputArray, tmp_targetArray, tmp_predArray, self.augPercent)
            else:
                tmp, targetArray = dataAugmentation(inputArray, tmp_targetArray, self.augPercent)
        else:
            tmp = inputArray
            targetArray = tmp_targetArray
            if self.isBase:
                predArray = tmp_predArray


        if self.isBase:
            inputArrayOut = torch.cat((torch.reshape(tmp, (8, 64, 64, 64)), predArray), 0)
        else:
            inputArrayOut = tmp
        targetArrayOut = targetArray

        return inputArrayOut, targetArrayOut


class DataClassLongVideo(Dataset):

    def __init__(self, dataPath, trainPercent, no_of_seq, isBase):
        self.dataPath = dataPath
        self.multipPercent = trainPercent
        self.no_of_seq = no_of_seq
        self.isBase = isBase
        self.multipPercent = 1 - trainPercent
        self.beginIndex = int(len(os.listdir(self.dataPath)) / 2 * (1 - self.multipPercent))
        self.mean, self.std = avg_mean_std(self.dataPath, 250)

        if self.isBase:
            self.arr_out = 10
        else:
            self.arr_out = 13

    def __len__(self):
        return int(len(os.listdir(self.dataPath)) / 2 * self.multipPercent * self.arr_out)

    def __getitem__(self, index):

        data_index = int(np.floor(index / self.arr_out))
        nt = index % self.arr_out + 3

        self.input = loadNpz(os.path.join(self.dataPath, "ex" + f"_{self.beginIndex + data_index}_" + "BP_guidewire_64x64x64x2x16.npz"))
        self.target = loadNpz(os.path.join(self.dataPath, "ex" + f"_{self.beginIndex + data_index}_" + "voxSTL_guidewire_64x64x64x16.npz"))
        if self.isBase:
            pred = readArray(os.path.join("C:\src\Predict_Results_12", "predict" + f"_{self.beginIndex + data_index - 20000}_" + "voxSTL_guidewire_64x64x64x13.raw"))

        
        inputArrayLoad = self.input[ nt:nt+self.no_of_seq+1, :, :, :, :].astype(np.float32)
        if self.isBase:
            targetArrayLoad = np.where(self.target[nt+self.no_of_seq, :, :, :] > 1e-5, 1, 0)
        else:
            targetArrayLoad = np.where(self.target[nt:nt+self.no_of_seq+1, :, :, :] > 1e-5, 1, 0)
        inputArrayLoad = ( inputArrayLoad - self.mean ) / self.std
        inputArray = torch.from_numpy(inputArrayLoad).type(torch.float32)
        tmp_targetArray = torch.from_numpy(targetArrayLoad).type(torch.float32)

        if self.isBase:
            predArrayLoad = np.where(pred[nt-self.no_of_seq:nt, :, :, :] > 1e-5, 1, 0)
            tmp_predArray = torch.from_numpy(predArrayLoad).type(torch.float32)
            if tmp_targetArray.dim() == 3:
                tmp_targetArray = tmp_targetArray.unsqueeze(0)

        tmp = inputArray
        targetArray = tmp_targetArray
        if self.isBase:
            predArray = tmp_predArray


        if self.isBase:
            inputArrayOut = torch.cat((torch.reshape(tmp, (8, 64, 64, 64)), predArray), 0)
        else:
            inputArrayOut = tmp
        targetArrayOut = targetArray

        nt = nt - 3
        return inputArrayOut, targetArrayOut, nt


def avg_mean_std(data_path, dataNo):
        mean = 0
        std = 0
        for ii in range(dataNo):
            currentDataPath = os.path.join(data_path, "ex" + f"_{ii}_" + "BP_guidewire_64x64x64x2x16.npz")
            data_cur = loadNpz(currentDataPath).astype(np.float32)
            mean += np.mean(data_cur, dtype=np.float32) / dataNo
            std += np.std(data_cur, dtype=np.float32) / dataNo
        return mean, std


def get_dataloaders(train_dataset, val_dataset, dataloader_workers: int = 3, batch_size: int = 8, prefetch_factor: int = 32):
    
    kwargs = {'pin_memory': True, 'num_workers': dataloader_workers, 'prefetch_factor': prefetch_factor}
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
        **kwargs
    )
    return {'train': train_loader, 'val': val_loader}


def get_dataloadersLongVideo(test_dataset, dataloader_workers: int = 3, batch_size: int = 8, prefetch_factor: int = 32):
    
    kwargs = {'pin_memory': True, 'num_workers': dataloader_workers, 'prefetch_factor': prefetch_factor}
    val_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    drop_last=False,
    shuffle=False,
    **kwargs
    )
    return {'val': val_loader}


def dice_coef1(y_true, y_pred):   
    y_true = torch.where(y_true < 0.5, 0., 1.)
    y_pred = torch.where(y_pred < 0.5, 0., 1.)
    y_true = torch.flatten(y_true)
    y_pred = torch.flatten(y_pred)
    intersection = torch.sum(y_true * y_pred)
    dice = (2. * intersection) / (torch.sum(y_true) + torch.sum(y_pred) + 1e-10)    
    return dice.type(torch.float32)


class DiceCoeff(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("dice", default=torch.tensor(0), dist_reduce_fx=None)

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        # y_pred, y_true = self._input_format(y_pred, y_true)
        y_true = torch.where(y_true < 0.5, 0., 1.)
        y_pred = torch.where(y_pred < 0.5, 0., 1.)
        y_true = torch.flatten(y_true)
        y_pred = torch.flatten(y_pred)
        intersection = torch.sum(y_true * y_pred)
        self.dice = (2. * intersection) / (torch.sum(y_true) + torch.sum(y_pred) + 1e-10) 

    def compute(self):
        return self.dice.type(torch.float32)

def dice_coef_loss(y_true, y_pred, smooth=1.):      
    y_true = torch.flatten(y_true)
    y_pred = torch.flatten(y_pred)
    intersection = torch.sum(torch.abs(y_true * y_pred))
    dice = (2. * intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)
    return (1 - dice).type(torch.float32)


def calc_loss(target, pred, loss_fn):
    loss = loss_fn(pred, target)
    return loss


def plot_training(total_loss):
    plt.plot(total_loss["loss0"], color='blue')
    plt.title("Loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(total_loss["loss1"], color='red')
    plt.plot(total_loss["loss2"], color='green')
    plt.plot(total_loss["loss3"], color='braun')
    plt.legend(['loss0', 'loss1', 'loss2', 'loss3'])
    plt.show()


class LossFunc:
    def __init__(self):
        pass


def train_model(model, dataloaders, use_cuda, epochs, optimizer, loss_fn, metric_fn, checkpoint_path_model = "current_best.pth", trained_epochs: int = 0, tb_writer = None, lossFunc=None, isBase=False):
    best_loss = 1e10
    total_loss = {key: [] for key in ['train', 'val']}
    loss_fn = loss_fn
    optimizer = optimizer
    since = time.time()

    metrics = MetricCollection([metric_fn])

    train_metrics = metrics.clone(prefix="train")
    val_metrics = metrics.clone(prefix="val")

    scaler = torch.cuda.amp.GradScaler()
    # iterate over all epochs
    for epoch in range(trained_epochs, epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0
            loss_list = {key: [] for key in ['loss0', 'loss1', 'loss2', 'loss3']}
            for dic in tqdm(dataloaders[phase], total=len(dataloaders[phase])):
                inputs, labels = dic

                if use_cuda:
                    inputs = inputs.to('cuda', dtype=torch.float32)  # [batch_size, in_channels, D, H, W]
                    labels = labels.to('cuda', dtype=torch.float32)
                             
                optimizer.zero_grad()  # zero the parameter gradients

                # forward pass: compute prediction and the loss btw prediction and true label
                # track history only in train
                with torch.set_grad_enabled(phase == 'train'):
                    with torch.autocast(device_type='cuda', dtype=torch.float32):
                        outputs = model(inputs)
                    
                        # output is binary [batch size, n_classes, D, H, W], target is class [batch size, D, H, W]
                        loss = 0
                        if isBase:
                            loss = calc_loss(labels, outputs, loss_fn)
                        else:
                            for i in range(outputs.shape[1]):
                                setattr(lossFunc, 'loss{}'.format(i), calc_loss(labels[:, i, :, :, :], outputs[:, i, :, :, :], loss_fn))
                                loss += getattr(lossFunc, 'loss{}'.format(i)) / outputs.shape[1]
                                if phase == 'train':
                                    loss_list[f'loss{i}'].append(getattr(lossFunc, 'loss{}'.format(i)) / outputs.shape[1])

                        # backward + optimize only if in training phase (no need for torch.no_grad in this training pass)
                        if phase == 'train':
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                if phase == 'train':
                    scaler.update()

                # statistics
                running_loss += loss * outputs.size(0)
                preds_cpu = outputs
                labels_cpu = labels
                if phase == "train":
                    train_metrics.update(preds_cpu, labels_cpu)
                elif phase == "val":
                    val_metrics.update(preds_cpu, labels_cpu)

            if phase == "train":
                computed_metrics = train_metrics.compute()
                train_metrics.reset()
            elif phase == "val":
                computed_metrics = val_metrics.compute()
                val_metrics.reset()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            computed_metrics[f"{phase}Loss"] = epoch_loss

            epoch_summary = f'Epoch {phase} : {epoch+1}'
            for k, v in computed_metrics.items():
              epoch_summary = f"{epoch_summary}\n\t{k} : {v:.6f}"

            print(epoch_summary)
            total_loss[phase].append(computed_metrics[f"{phase}Loss"].item())

        
            # Display metrics in Tensorboard
            if tb_writer is not None:
                for item in ["Loss", "DiceCoeff"]:
                    tb_writer[0].add_scalar(f"{item}/{phase}", computed_metrics[f"{phase}{item}"], epoch)
                #for i in range(4):
                    #tb_writer[i+1].add_scalar(f"LossSequences/{phase}", getattr(lossFunc, 'loss{}'.format(i)) / outputs.shape[1], epoch)

            # save the model weights in validation phase 
            if phase == 'val':
                if epoch_loss < best_loss:
                    print(f"saving best model to {checkpoint_path_model}")
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), checkpoint_path_model)

        # Display total time
        time_elapsed = time.time() - since
        print('Total time elapsed: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # plot_training(total_loss)

    # load best model weights
    model.load_state_dict(torch.load(checkpoint_path_model))

    return model


def test_model(model, dataloaders, use_cuda, optimizer, loss_fn, metric_fn, resultFilePath = "", tb_writer = None, outStepsNo=4, lossFunc=None, isBase=False):
    loss_fn = loss_fn
    optimizer = optimizer
    batch_size = dataloaders['val'].batch_size
    since = time.time()

    metrics = MetricCollection([metric_fn])

    test_metrics = metrics.clone(prefix="test")
    model.eval()

    # iterate over all epochs
    with torch.no_grad():

        running_loss = 0
        for x, dic in tqdm(enumerate(dataloaders['val']), total=len(dataloaders['val'])):
            inputs, labels = dic

            if use_cuda:
                inputs = inputs.to('cuda', dtype=torch.float32)  # [batch_size, in_channels, D, H, W]
                labels = labels.to('cuda', dtype=torch.float32)
                            
            optimizer.zero_grad()  # zero the parameter gradients

            # forward pass: compute prediction and the loss btw prediction and true label
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                outputs = model(inputs)
            
                # output is binary [batch size, n_classes, D, H, W], target is class [batch size, D, H, W]
                loss = 0
                if isBase:
                    loss = calc_loss(labels, outputs, loss_fn)
                    setattr(lossFunc, 'metric{}'.format(0), dice_coef1(outputs, labels))
                else:
                    for i in range(outputs.shape[1]):
                        setattr(lossFunc, 'loss{}'.format(i), calc_loss(labels[:, i, :, :, :], outputs[:, i, :, :, :], loss_fn))
                        loss += getattr(lossFunc, 'loss{}'.format(i)) / outputs.shape[1]

                        setattr(lossFunc, 'metric{}'.format(i), dice_coef1(outputs[:, i, :, :, :], labels[:, i, :, :, :]))

            # statistics
            running_loss += loss * outputs.size(0)
            test_metrics.update(outputs, labels)

            # for x in range(startIndex, startIndex + len(dataloaders['val'])):
            for i in range(batch_size):
                file_name = "predict_" + str(x*batch_size + i) + f"_voxSTL_guidewire_64x64x64x{outStepsNo}.raw"
                save_path = os.path.join(resultFilePath, file_name)
                writeArray(outputs[i, :, :, :, :].detach().cpu().numpy().astype(np.float32), save_path)

        computed_metrics = test_metrics.compute()
        ##  test_metrics.reset()

        epoch_loss = running_loss / len(dataloaders['val'].dataset)
        computed_metrics["testLoss"] = epoch_loss

        for i in range(outStepsNo):
            computed_metrics[f"testDiceCoeff{i}"] = getattr(lossFunc, 'metric{}'.format(i))

    
        # Display metrics in Tensorboard
        if tb_writer is not None:
            for item in ["Loss"]:
                tb_writer[0].add_scalar(f"{item}/test", computed_metrics[f"test{item}"], 0)

            for i in range(outStepsNo):
                tb_writer[0].add_scalar("DiceCoeff/test", computed_metrics[f"testDiceCoeff{i}"], i+3)


    # Display total time
    time_elapsed = time.time() - since
    print('Total time elapsed: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


def test_modelLongVideo(model, dataloaders, use_cuda, resultFilePath = "", outStepsNo=13, isBase=True):
    since = time.time()
    model.eval()

    # iterate over all epochs
    with torch.no_grad():
        
        out_all = torch.zeros([outStepsNo, 64, 64, 64])
        index = 0
        for dic in tqdm(dataloaders['val'], total=len(dataloaders['val'])):
            inputs, labels, nt = dic

            if use_cuda:
                inputs = inputs.to('cuda', dtype=torch.float32)  # [batch_size, in_channels, D, H, W]
                labels = labels.to('cuda', dtype=torch.float32)
                out_all = out_all.to('cuda', dtype=torch.float32)

            # forward pass: compute prediction and save the prediction at the specified location
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                outputs = model(inputs)

                if isBase:
                    out_all[nt, :, :, :] = outputs[0, 0, :, :, :].unsqueeze(0)
                else:
                    out_all[nt, :, :, :] = outputs[0, -1, :, :, :].unsqueeze(0)
                if nt % outStepsNo == outStepsNo - 1:
                    file_name = "predict_" + str(index) + f"_voxSTL_guidewire_64x64x64x{outStepsNo}.raw"
                    save_path = os.path.join(resultFilePath, file_name)
                    writeArray(out_all.detach().cpu().numpy().astype(np.float32), save_path)
                    out_all = out_all * 0
                    index = index + 1


    # Display total time
    time_elapsed = time.time() - since
    print('Total time elapsed: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))



if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # Choose the experiment number
    exp_no = 3

    # Make true for testing
    isTest = False
    # Make true for making a long video of predictions from the inputs with all time sequences
    isLongVideo = False
    outStepsNo = 10
    # Make true for resuming training
    isResume = False
    trained_epochs = 0

    # Define modifiable training hyperparameters.
    epochs = 15
    batch_size = 4
    learning_rate = 1e-4
    weight_decay = 1e-6
    no_of_old_seq = 3
    # 70% training and 30% validation
    train_valSplit = 0.7
    # how often should the data augmentation be applied to the input data
    dataAugProb = 0.5

    # Dataset file paths to take the data. Tensorboard and best model file paths that we want to save.
    # The input data inside the files should be in the [time sequences, channels, depth, height, width] format. Depth, width, height order can be changed.
    # The target data should be in the [time sequences, depth, height, width] format. Depth, width, height order can be changed.
    train_valDataset = 'C:/data/2022-11-21_guidewire_noBin_40000x16x64_p64_SO1.0E+06_cem/Results'
    tensorboardFilePath = f"C:/src/logs_torch/Exp{exp_no}_limitlessSeq"
    bestModelFilePath = f"C:/src/best_models/exp{exp_no}_limitlessSeq.pth"
    predictionFilePath = f"C:/src/Predict_Results_Exp{exp_no}" # only for test

    # Experiments
    if exp_no == 0:
        model = UNet(filterList=[11, 16, 32, 64, 128], step=1, kernel_sizes=[3, 3, 3, 3, 3], out_classes=1, device=device)
    elif exp_no == 1:
        model = UNetRNN(filterList=[2, 16, 32, 64, 128], step=1, kernel_sizes=[3, 3, 3, 3, 3], out_classes=1, device=device, resolutionStage=1, convType='LSTM', isPeep=False)
    elif exp_no == 2:
        model = UNetRNN(filterList=[2, 16, 32, 64, 128], step=2, kernel_sizes=[3, 3, 3, 3, 3], out_classes=1, device=device, resolutionStage=1, convType='LSTM', isPeep=False)
    elif exp_no == 3:
        model = UNetRNN(filterList=[2, 16, 32, 64, 128], step=1, kernel_sizes=[3, 3, 3, 3, 3], out_classes=1, device=device, resolutionStage=1, convType='Mix', isPeep=False)
    elif exp_no == 4:
        model = UNetRNN(filterList=[2, 16, 32, 64, 128], step=1, kernel_sizes=[3, 3, 3, 3, 3], out_classes=1, device=device, resolutionStage=1, convType='LSTM', isPeep=True)
    elif exp_no == 5:
        model = UNetRNN(filterList=[2, 16, 32, 64, 128], step=1, kernel_sizes=[3, 3, 3, 3, 3], out_classes=1, device=device, resolutionStage=2, convType='LSTM', isPeep=False)
    elif exp_no == 6:
        model = UNetRNN(filterList=[2, 16, 32, 64, 128], step=2, kernel_sizes=[3, 3, 3, 3, 3], out_classes=1, device=device, resolutionStage=1, convType='GRU', isPeep=False)
    else:
        print( " Wrong experiment number, check the exp_no again!! " )
    model = model.to(device, dtype=torch.float)

    # Dataset, optimizer, metric and loss functions, tensorboard writer
    data_train = DataClass(train_valDataset, 'train', train_valSplit, no_of_old_seq, dataAugProb, isBase = exp_no==0)
    data_val = DataClass(train_valDataset, 'test' if isTest else 'val', train_valSplit, no_of_old_seq, 0.0, isBase = exp_no==0)
    dataloaders = get_dataloaders(data_train, data_val, batch_size=batch_size)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    dice_coef = DiceCoeff()
    writer = [SummaryWriter(tensorboardFilePath)] 
    #, SummaryWriter("C:/src/logs_torch/exp1_0_1"), SummaryWriter("C:/src/logs_torch/exp1_1_1"), SummaryWriter("C:/src/logs_torch/exp1_2_1"), SummaryWriter("C:/src/logs_torch/exp1_3_1")]
    loss_func = LossFunc()

    if isTest or isLongVideo:
        model.load_state_dict(torch.load(bestModelFilePath))

    if isLongVideo:
        data_test = DataClassLongVideo(train_valDataset, 0.9995, no_of_old_seq, isBase = exp_no==0)
        dataloaderLongVideo = get_dataloadersLongVideo(test_dataset=data_test, batch_size=batch_size)
    
    if isTest:
        test_model(
        model = model,
        dataloaders = dataloaders,
        use_cuda = torch.cuda.is_available(),
        optimizer = optimizer,
        loss_fn = dice_coef_loss,
        metric_fn = dice_coef,
        resultFilePath = predictionFilePath,
        tb_writer = writer,
        outStepsNo = 1 if exp_no==0 else outStepsNo,
        lossFunc = loss_func,
        isBase= exp_no==0
        )
    elif isLongVideo:
        test_modelLongVideo(
        model = model,
        dataloaders = dataloaderLongVideo,
        use_cuda = torch.cuda.is_available(),
        resultFilePath = predictionFilePath,
        outStepsNo = outStepsNo,
        isBase= exp_no==0
        )
    else:
        if isResume:
            model.load_state_dict(torch.load(bestModelFilePath))

        # Fit the model to the training data.
        model = train_model(
            model = model,
            dataloaders = dataloaders,
            use_cuda = torch.cuda.is_available(),
            epochs = epochs,
            optimizer = optimizer,
            checkpoint_path_model = bestModelFilePath,
            loss_fn = dice_coef_loss,
            metric_fn = dice_coef,
            tb_writer = writer,
            trained_epochs = trained_epochs,
            lossFunc = loss_func,
            isBase = exp_no==0
        )