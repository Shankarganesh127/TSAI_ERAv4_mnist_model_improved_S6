# 🧠 Lightweight MNIST CNN Models: model_v0.py vs model_v1.py

This project implements and compares two compact Convolutional Neural Networks (CNNs) for the MNIST handwritten digits dataset. Both models are designed to be highly parameter-efficient while achieving >99.4% accuracy.

---

## 🚀 Model Highlights

| Model      | Parameters | Test Accuracy (15 epochs) | Key Features |
|------------|------------|--------------------------|--------------|
| model_v0.py| 18,214     | 99.44%                   | GAP, FC, BatchNorm, MaxPool |
| model_v1.py| 10,970     | 99.40%                   | Deeper, Dropout, Fewer Params |

---

## 🏗️ Model Architectures

### model_v0.py
- **Structure:** 3 convolutional blocks (channels: 4→8→16→32→40), each with BatchNorm and ReLU, MaxPooling after each block, ends with Global Average Pooling and a fully connected layer.
- **Output:** Uses `F.log_softmax` for NLLLoss.
- **Parameters:** ~18,214.

<details>
<summary>Click to expand model_v0.py architecture</summary>

```text
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 4, 28, 28]              36
       BatchNorm2d-2            [-1, 4, 28, 28]               8
              ReLU-3            [-1, 4, 28, 28]               0
            Conv2d-4            [-1, 8, 28, 28]             288
       BatchNorm2d-5            [-1, 8, 28, 28]              16
              ReLU-6            [-1, 8, 28, 28]               0
         MaxPool2d-7            [-1, 8, 14, 14]               0
            Conv2d-8           [-1, 16, 14, 14]           1,152
       BatchNorm2d-9           [-1, 16, 14, 14]              32
             ReLU-10           [-1, 16, 14, 14]               0
           Conv2d-11           [-1, 32, 14, 14]           4,608
      BatchNorm2d-12           [-1, 32, 14, 14]              64
             ReLU-13           [-1, 32, 14, 14]               0
        MaxPool2d-14             [-1, 32, 7, 7]               0
           Conv2d-15             [-1, 40, 7, 7]          11,520
      BatchNorm2d-16             [-1, 40, 7, 7]              80
             ReLU-17             [-1, 40, 7, 7]               0
AdaptiveAvgPool2d-18             [-1, 40, 1, 1]               0
           Linear-19                   [-1, 10]             410
================================================================
Total params: 18,214
Trainable params: 18,214
Non-trainable params: 0
----------------------------------------------------------------
```
</details>

### model_v1.py
- **Structure:** Deeper, more modular. Initial conv (1→10), then blocks with 10→10→20, 20→10→10→20, 20→10→10, with BatchNorm, ReLU, Dropout, and MaxPool. Final conv uses 7x7 kernel to reduce to 1x1.
- **Output:** No explicit GAP or FC; output is flattened. (You may want to add `log_softmax` in the forward or loss function.)
- **Parameters:** ~10,970 (fewer than v0).

<details>
<summary>Click to expand model_v1.py architecture</summary>

```text
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 10, 26, 26]              90
       BatchNorm2d-2           [-1, 10, 26, 26]              20
              ReLU-3           [-1, 10, 26, 26]               0
            Conv2d-4           [-1, 10, 24, 24]             900
       BatchNorm2d-5           [-1, 10, 24, 24]              20
              ReLU-6           [-1, 10, 24, 24]               0
            Conv2d-7           [-1, 20, 22, 22]           1,800
       BatchNorm2d-8           [-1, 20, 22, 22]              40
              ReLU-9           [-1, 20, 22, 22]               0
        Dropout2d-10           [-1, 20, 22, 22]               0
        MaxPool2d-11           [-1, 20, 11, 11]               0
           Conv2d-12           [-1, 10, 11, 11]             200
      BatchNorm2d-13           [-1, 10, 11, 11]              20
             ReLU-14           [-1, 10, 11, 11]               0
           Conv2d-15             [-1, 10, 9, 9]             900
      BatchNorm2d-16             [-1, 10, 9, 9]              20
             ReLU-17             [-1, 10, 9, 9]               0
           Conv2d-18             [-1, 20, 7, 7]           1,800
      BatchNorm2d-19             [-1, 20, 7, 7]              40
             ReLU-20             [-1, 20, 7, 7]               0
        Dropout2d-21             [-1, 20, 7, 7]               0
           Conv2d-22             [-1, 10, 7, 7]             200
      BatchNorm2d-23             [-1, 10, 7, 7]              20
             ReLU-24             [-1, 10, 7, 7]               0
           Conv2d-25             [-1, 10, 1, 1]           4,900
================================================================
Total params: 10,970
Trainable params: 10,970
Non-trainable params: 0
----------------------------------------------------------------
```
</details>

---

## 📊 Training Results: Side-by-Side Epoch Table

| Epoch | v0 Train Acc | v0 Test Acc | v1 Train Acc | v1 Test Acc |
|-------|--------------|-------------|--------------|-------------|
| 1     | 90.47%       | 97.43%      | 90.33%       | 98.21%      |
| 2     | 97.43%       | 98.75%      | 95.19%       | 98.79%      |
| 3     | 97.92%       | 98.94%      | 96.19%       | 98.75%      |
| 4     | 98.07%       | 98.90%      | 96.42%       | 98.99%      |
| 5     | 98.40%       | 99.04%      | 96.68%       | 99.12%      |
| 6     | 98.52%       | 98.84%      | 96.81%       | 98.78%      |
| 7     | 98.90%       | 99.35%      | 97.58%       | 99.27%      |
| 8     | 98.96%       | 99.36%      | 97.70%       | 99.30%      |
| 9     | 99.00%       | 99.34%      | 97.89%       | 99.29%      |
| 10    | 99.03%       | 99.35%      | 97.89%       | 99.28%      |
| 11    | 99.04%       | 99.38%      | 97.92%       | 99.40%      |
| 12    | 99.01%       | 99.44%      | 97.95%       | 99.39%      |
| 13    | 99.07%       | 99.41%      | 98.00%       | 99.39%      |
| 14    | 99.14%       | 99.38%      | 97.99%       | 99.38%      |
| 15    | 99.05%       | 99.41%      | 98.07%       | 99.37%      |

---

## 📈 Visualizing Training

You can plot the accuracy and loss curves for both models using matplotlib:

```python
import matplotlib.pyplot as plt

# Example: Plotting accuracy curves
epochs = list(range(1, 16))
v0_train_acc = [90.47,97.43,97.92,98.07,98.40,98.52,98.90,98.96,99.00,99.03,99.04,99.01,99.07,99.14,99.05]
v0_test_acc  = [97.43,98.75,98.94,98.90,99.04,98.84,99.35,99.36,99.34,99.35,99.38,99.44,99.41,99.38,99.41]
v1_train_acc = [90.33,95.19,96.19,96.42,96.68,96.81,97.58,97.70,97.89,97.89,97.92,97.95,98.00,97.99,98.07]
v1_test_acc  = [98.21,98.79,98.75,98.99,99.12,98.78,99.27,99.30,99.29,99.28,99.40,99.39,99.39,99.38,99.37]

plt.figure(figsize=(10,5))
plt.plot(epochs, v0_train_acc, label='v0 Train Acc', marker='o')
plt.plot(epochs, v0_test_acc, label='v0 Test Acc', marker='o')
plt.plot(epochs, v1_train_acc, label='v1 Train Acc', marker='x')
plt.plot(epochs, v1_test_acc, label='v1 Test Acc', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Train/Test Accuracy per Epoch')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 📝 Summary

- **model_v0.py**: Simpler, more parameters, achieves high accuracy quickly.
- **model_v1.py**: Deeper, more regularization, fewer parameters, matches v0's accuracy.
- Both models are highly efficient for MNIST and demonstrate the power of compact CNNs.

---

# Lightweight MNIST CNN Model (18,214 Parameters)

This project implements a **compact Convolutional Neural Network (CNN)** for the MNIST handwritten digits dataset.  
The model is optimized to stay **under 20,000 trainable parameters** while still achieving **>99.4% accuracy** on the MNIST test set.

---

## 🚀 Model Highlights **model_v0.py**
- **Dataset**: [MNIST](http://yann.lecun.com/exdb/mnist/) (28x28 grayscale digits, 10 classes)  
- **Total Parameters**: **18,214** (all trainable)  
- **Test Accuracy**: **99.44%** (within 15 epochs)  
- **Efficiency**: Uses **Batch Normalization + Global Average Pooling** for stability and parameter reduction  

---

## 📊 Training Results (Epoch by Epoch)

| Epoch | Train Loss | Train Accuracy | Test Loss | Test Accuracy |
|-------|------------|----------------|-----------|---------------|
| 1     | 0.0049     | 90.47%         | 0.0001    | 97.43%        |
| 2     | 0.0014     | 97.43%         | 0.0000    | 98.75%        |
| 3     | 0.0011     | 97.92%         | 0.0000    | 98.94%        |
| 4     | 0.0009     | 98.07%         | 0.0000    | 98.90%        |
| 5     | 0.0008     | 98.40%         | 0.0000    | 99.04%        |
| 6     | 0.0008     | 98.52%         | 0.0000    | 98.84%        |
| 7     | 0.0006     | 98.90%         | 0.0000    | 99.35%        |
| 8     | 0.0005     | 98.96%         | 0.0000    | 99.36%        |
| 9     | 0.0005     | 99.00%         | 0.0000    | 99.34%        |
| 10    | 0.0005     | 99.03%         | 0.0000    | 99.35%        |
| 11    | 0.0005     | 99.04%         | 0.0000    | 99.38%        |
| 12    | 0.0005     | 99.01%         | 0.0000    | 99.44% ✅     |
| 13    | 0.0005     | 99.07%         | 0.0000    | 99.41%        |
| 14    | 0.0005     | 99.14%         | 0.0000    | 99.38%        |
| 15    | 0.0005     | 99.05%         | 0.0000    | 99.41%        |

---

## 🏗️ Model Architecture **model_v0.py**

```text
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 4, 28, 28]              36
       BatchNorm2d-2            [-1, 4, 28, 28]               8
              ReLU-3            [-1, 4, 28, 28]               0
            Conv2d-4            [-1, 8, 28, 28]             288
       BatchNorm2d-5            [-1, 8, 28, 28]              16
              ReLU-6            [-1, 8, 28, 28]               0
         MaxPool2d-7            [-1, 8, 14, 14]               0
            Conv2d-8           [-1, 16, 14, 14]           1,152
       BatchNorm2d-9           [-1, 16, 14, 14]              32
             ReLU-10           [-1, 16, 14, 14]               0
           Conv2d-11           [-1, 32, 14, 14]           4,608
      BatchNorm2d-12           [-1, 32, 14, 14]              64
             ReLU-13           [-1, 32, 14, 14]               0
        MaxPool2d-14             [-1, 32, 7, 7]               0
           Conv2d-15             [-1, 40, 7, 7]          11,520
      BatchNorm2d-16             [-1, 40, 7, 7]              80
             ReLU-17             [-1, 40, 7, 7]               0
AdaptiveAvgPool2d-18             [-1, 40, 1, 1]               0
           Linear-19                   [-1, 10]             410
================================================================
Total params: 18,214
Trainable params: 18,214
Non-trainable params: 0
----------------------------------------------------------------

**model_v1.py**
Improvements:
1) Reducing model parameter using christmas tree structure 

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 10, 26, 26]              90
       BatchNorm2d-2           [-1, 10, 26, 26]              20
              ReLU-3           [-1, 10, 26, 26]               0
            Conv2d-4           [-1, 10, 24, 24]             900
       BatchNorm2d-5           [-1, 10, 24, 24]              20
              ReLU-6           [-1, 10, 24, 24]               0
            Conv2d-7           [-1, 20, 22, 22]           1,800
       BatchNorm2d-8           [-1, 20, 22, 22]              40
              ReLU-9           [-1, 20, 22, 22]               0
        Dropout2d-10           [-1, 20, 22, 22]               0
        MaxPool2d-11           [-1, 20, 11, 11]               0
           Conv2d-12           [-1, 10, 11, 11]             200
      BatchNorm2d-13           [-1, 10, 11, 11]              20
             ReLU-14           [-1, 10, 11, 11]               0
           Conv2d-15             [-1, 10, 9, 9]             900
      BatchNorm2d-16             [-1, 10, 9, 9]              20
             ReLU-17             [-1, 10, 9, 9]               0
           Conv2d-18             [-1, 20, 7, 7]           1,800
      BatchNorm2d-19             [-1, 20, 7, 7]              40
             ReLU-20             [-1, 20, 7, 7]               0
        Dropout2d-21             [-1, 20, 7, 7]               0
           Conv2d-22             [-1, 10, 7, 7]             200
      BatchNorm2d-23             [-1, 10, 7, 7]              20
             ReLU-24             [-1, 10, 7, 7]               0
           Conv2d-25             [-1, 10, 1, 1]           4,900
================================================================
Total params: 10,970
Trainable params: 10,970
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.69
Params size (MB): 0.04
Estimated Total Size (MB): 0.73
----------------------------------------------------------------
Epoch 1
Train Loss=0.0047 Accuracy=90.33: 100%|██████████| 938/938 [00:28<00:00, 32.77it/s]

Test set: Average loss: 0.0001, Accuracy: 9821/10000 (98.21%)

Epoch 2
Train Loss=0.0024 Accuracy=95.19: 100%|██████████| 938/938 [00:28<00:00, 32.99it/s]

Test set: Average loss: 0.0000, Accuracy: 9879/10000 (98.79%)

Epoch 3
Train Loss=0.0019 Accuracy=96.19: 100%|██████████| 938/938 [00:28<00:00, 32.55it/s]

Test set: Average loss: 0.0000, Accuracy: 9875/10000 (98.75%)

Epoch 4
Train Loss=0.0019 Accuracy=96.42: 100%|██████████| 938/938 [00:28<00:00, 32.86it/s]

Test set: Average loss: 0.0000, Accuracy: 9899/10000 (98.99%)

Epoch 5
Train Loss=0.0017 Accuracy=96.68: 100%|██████████| 938/938 [00:28<00:00, 33.01it/s]

Test set: Average loss: 0.0000, Accuracy: 9912/10000 (99.12%)

Epoch 6
Train Loss=0.0016 Accuracy=96.81: 100%|██████████| 938/938 [00:29<00:00, 32.16it/s]

Test set: Average loss: 0.0000, Accuracy: 9878/10000 (98.78%)

Epoch 7
Train Loss=0.0012 Accuracy=97.58: 100%|██████████| 938/938 [00:28<00:00, 32.94it/s]

Test set: Average loss: 0.0000, Accuracy: 9927/10000 (99.27%)

Epoch 8
Train Loss=0.0011 Accuracy=97.70: 100%|██████████| 938/938 [00:29<00:00, 32.34it/s]

Test set: Average loss: 0.0000, Accuracy: 9930/10000 (99.30%)

Epoch 9
Train Loss=0.0011 Accuracy=97.89: 100%|██████████| 938/938 [00:28<00:00, 32.59it/s]

Test set: Average loss: 0.0000, Accuracy: 9929/10000 (99.29%)

Epoch 10
Train Loss=0.0011 Accuracy=97.89: 100%|██████████| 938/938 [00:28<00:00, 32.42it/s]

Test set: Average loss: 0.0000, Accuracy: 9928/10000 (99.28%)

Epoch 11
Train Loss=0.0011 Accuracy=97.92: 100%|██████████| 938/938 [00:28<00:00, 33.34it/s]

Test set: Average loss: 0.0000, Accuracy: 9940/10000 (99.40%)

Epoch 12
Train Loss=0.0010 Accuracy=97.95: 100%|██████████| 938/938 [00:28<00:00, 32.98it/s]

Test set: Average loss: 0.0000, Accuracy: 9939/10000 (99.39%)

Epoch 13
Train Loss=0.0010 Accuracy=98.00: 100%|██████████| 938/938 [00:28<00:00, 33.48it/s]

Test set: Average loss: 0.0000, Accuracy: 9939/10000 (99.39%)

Epoch 14
Train Loss=0.0010 Accuracy=97.99: 100%|██████████| 938/938 [00:28<00:00, 32.98it/s]

Test set: Average loss: 0.0000, Accuracy: 9938/10000 (99.38%)

Epoch 15
Train Loss=0.0010 Accuracy=98.07: 100%|██████████| 938/938 [00:29<00:00, 31.69it/s]

Test set: Average loss: 0.0000, Accuracy: 9937/10000 (99.37%)



** model_v2.py **

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 4, 28, 28]              36
       BatchNorm2d-2            [-1, 4, 28, 28]               8
              ReLU-3            [-1, 4, 28, 28]               0
            Conv2d-4            [-1, 8, 28, 28]             288
       BatchNorm2d-5            [-1, 8, 28, 28]              16
              ReLU-6            [-1, 8, 28, 28]               0
         MaxPool2d-7            [-1, 8, 14, 14]               0
            Conv2d-8           [-1, 10, 14, 14]             720
       BatchNorm2d-9           [-1, 10, 14, 14]              20
             ReLU-10           [-1, 10, 14, 14]               0
        Dropout2d-11           [-1, 10, 14, 14]               0
           Conv2d-12           [-1, 12, 14, 14]           1,080
      BatchNorm2d-13           [-1, 12, 14, 14]              24
             ReLU-14           [-1, 12, 14, 14]               0
        MaxPool2d-15             [-1, 12, 7, 7]               0
           Conv2d-16              [-1, 8, 7, 7]              96
      BatchNorm2d-17              [-1, 8, 7, 7]              16
             ReLU-18              [-1, 8, 7, 7]               0
           Conv2d-19             [-1, 12, 7, 7]             864
      BatchNorm2d-20             [-1, 12, 7, 7]              24
             ReLU-21             [-1, 12, 7, 7]               0
           Conv2d-22             [-1, 10, 7, 7]             120
      BatchNorm2d-23             [-1, 10, 7, 7]              20
             ReLU-24             [-1, 10, 7, 7]               0
           Conv2d-25             [-1, 10, 5, 5]             900
        AvgPool2d-26             [-1, 10, 1, 1]               0
          Flatten-27                   [-1, 10]               0
================================================================
Total params: 4,232
Trainable params: 4,232
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.38
Params size (MB): 0.02
Estimated Total Size (MB): 0.40
----------------------------------------------------------------
Epoch 1
Train Loss=0.0030 Accuracy=88.06: 100%|██████████| 469/469 [00:58<00:00,  8.00it/s]

Test set: Average loss: 0.2505, Accuracy: 9185/10000 (91.85%)

Epoch 2
Train Loss=0.0009 Accuracy=96.39: 100%|██████████| 469/469 [00:59<00:00,  7.94it/s]

Test set: Average loss: 0.0691, Accuracy: 9783/10000 (97.83%)

Epoch 3
Train Loss=0.0007 Accuracy=97.13: 100%|██████████| 469/469 [00:59<00:00,  7.88it/s]

Test set: Average loss: 0.0511, Accuracy: 9844/10000 (98.44%)

Epoch 4
Train Loss=0.0007 Accuracy=97.41: 100%|██████████| 469/469 [00:58<00:00,  8.00it/s]

Test set: Average loss: 0.0468, Accuracy: 9855/10000 (98.55%)

Epoch 5
Train Loss=0.0006 Accuracy=97.76: 100%|██████████| 469/469 [00:59<00:00,  7.88it/s]

Test set: Average loss: 0.0371, Accuracy: 9878/10000 (98.78%)

Epoch 6
Train Loss=0.0006 Accuracy=97.76: 100%|██████████| 469/469 [00:59<00:00,  7.92it/s]

Test set: Average loss: 0.0418, Accuracy: 9858/10000 (98.58%)

Epoch 7
Train Loss=0.0004 Accuracy=98.31: 100%|██████████| 469/469 [00:59<00:00,  7.85it/s]

Test set: Average loss: 0.0274, Accuracy: 9911/10000 (99.11%)

Epoch 8
Train Loss=0.0004 Accuracy=98.49: 100%|██████████| 469/469 [00:59<00:00,  7.82it/s]

Test set: Average loss: 0.0271, Accuracy: 9910/10000 (99.10%)

Epoch 9
Train Loss=0.0004 Accuracy=98.51: 100%|██████████| 469/469 [00:59<00:00,  7.91it/s]

Test set: Average loss: 0.0266, Accuracy: 9913/10000 (99.13%)

Epoch 10
Train Loss=0.0004 Accuracy=98.46: 100%|██████████| 469/469 [00:59<00:00,  7.93it/s]

Test set: Average loss: 0.0258, Accuracy: 9907/10000 (99.07%)

Epoch 11
Train Loss=0.0004 Accuracy=98.61: 100%|██████████| 469/469 [00:59<00:00,  7.91it/s]

Test set: Average loss: 0.0270, Accuracy: 9911/10000 (99.11%)

Epoch 12
Train Loss=0.0004 Accuracy=98.60: 100%|██████████| 469/469 [00:59<00:00,  7.85it/s]

Test set: Average loss: 0.0257, Accuracy: 9916/10000 (99.16%)

Epoch 13
Train Loss=0.0004 Accuracy=98.57: 100%|██████████| 469/469 [00:59<00:00,  7.86it/s]

Test set: Average loss: 0.0259, Accuracy: 9915/10000 (99.15%)

Epoch 14
Train Loss=0.0004 Accuracy=98.58: 100%|██████████| 469/469 [00:58<00:00,  7.98it/s]

Test set: Average loss: 0.0255, Accuracy: 9915/10000 (99.15%)

Epoch 15
Train Loss=0.0004 Accuracy=98.63: 100%|██████████| 469/469 [00:59<00:00,  7.94it/s]

Test set: Average loss: 0.0254, Accuracy: 9916/10000 (99.16%)

--- Model Architecture Checks ---
Total Parameter Count in Model: 4232

Layer-wise Parameter Details (in model order):
--------------------------------------------------------------------------------

Block: input_conv (nn.Sequential)
  input_conv.0 (Conv2d)                    | Params:     36 | Convolution: 1 input channels, 4 output channels, kernel size (3, 3), bias False
  input_conv.1 (BatchNorm2d)               | Params:      8 | BatchNorm: 4 features, affine True
  input_conv.2 (ReLU)                      | Params:      0 | Activation: ReLU (no parameters)
  input_conv.3 (Conv2d)                    | Params:    288 | Convolution: 4 input channels, 8 output channels, kernel size (3, 3), bias False
  input_conv.4 (BatchNorm2d)               | Params:     16 | BatchNorm: 8 features, affine True
  input_conv.5 (ReLU)                      | Params:      0 | Activation: ReLU (no parameters)
  input_conv.6 (MaxPool2d)                 | Params:      0 | MaxPooling: kernel size 2, stride 2

Block: feature_block (nn.Sequential)
  feature_block.0 (Conv2d)                 | Params:    720 | Convolution: 8 input channels, 10 output channels, kernel size (3, 3), bias False
  feature_block.1 (BatchNorm2d)            | Params:     20 | BatchNorm: 10 features, affine True
  feature_block.2 (ReLU)                   | Params:      0 | Activation: ReLU (no parameters)
  feature_block.3 (Dropout2d)              | Params:      0 | 
  feature_block.4 (Conv2d)                 | Params:   1080 | Convolution: 10 input channels, 12 output channels, kernel size (3, 3), bias False
  feature_block.5 (BatchNorm2d)            | Params:     24 | BatchNorm: 12 features, affine True
  feature_block.6 (ReLU)                   | Params:      0 | Activation: ReLU (no parameters)
  feature_block.7 (MaxPool2d)              | Params:      0 | MaxPooling: kernel size 2, stride 2

Block: pattern_block (nn.Sequential)
  pattern_block.0 (Conv2d)                 | Params:     96 | Convolution: 12 input channels, 8 output channels, kernel size (1, 1), bias False
  pattern_block.1 (BatchNorm2d)            | Params:     16 | BatchNorm: 8 features, affine True
  pattern_block.2 (ReLU)                   | Params:      0 | Activation: ReLU (no parameters)
  pattern_block.3 (Conv2d)                 | Params:    864 | Convolution: 8 input channels, 12 output channels, kernel size (3, 3), bias False
  pattern_block.4 (BatchNorm2d)            | Params:     24 | BatchNorm: 12 features, affine True
  pattern_block.5 (ReLU)                   | Params:      0 | Activation: ReLU (no parameters)

Block: classifier (nn.Sequential)
  classifier.0 (Conv2d)                    | Params:    120 | Convolution: 12 input channels, 10 output channels, kernel size (1, 1), bias False
  classifier.1 (BatchNorm2d)               | Params:     20 | BatchNorm: 10 features, affine True
  classifier.2 (ReLU)                      | Params:      0 | Activation: ReLU (no parameters)
  classifier.3 (Conv2d)                    | Params:    900 | Convolution: 10 input channels, 10 output channels, kernel size (3, 3), bias False
  classifier.4 (AvgPool2d)                 | Params:      0 | 
  classifier.5 (Flatten)                   | Params:      0 | 
--------------------------------------------------------------------------------

Summary:
BatchNorm2d layers used: 7
Dropout layers used: 0
Fully Connected (Linear) layers used: 0
Global Average Pooling layers used: 0
---------------------------------


** model_v3.py ** 

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              72
       BatchNorm2d-2            [-1, 8, 28, 28]              16
              ReLU-3            [-1, 8, 28, 28]               0
            Conv2d-4           [-1, 10, 28, 28]             720
       BatchNorm2d-5           [-1, 10, 28, 28]              20
              ReLU-6           [-1, 10, 28, 28]               0
         MaxPool2d-7           [-1, 10, 14, 14]               0
            Conv2d-8           [-1, 10, 14, 14]             900
       BatchNorm2d-9           [-1, 10, 14, 14]              20
             ReLU-10           [-1, 10, 14, 14]               0
        Dropout2d-11           [-1, 10, 14, 14]               0
           Conv2d-12           [-1, 16, 14, 14]           1,440
      BatchNorm2d-13           [-1, 16, 14, 14]              32
             ReLU-14           [-1, 16, 14, 14]               0
        MaxPool2d-15             [-1, 16, 7, 7]               0
           Conv2d-16             [-1, 10, 7, 7]             160
      BatchNorm2d-17             [-1, 10, 7, 7]              20
             ReLU-18             [-1, 10, 7, 7]               0
           Conv2d-19             [-1, 16, 7, 7]           1,440
      BatchNorm2d-20             [-1, 16, 7, 7]              32
             ReLU-21             [-1, 16, 7, 7]               0
           Conv2d-22             [-1, 10, 5, 5]           1,440
      BatchNorm2d-23             [-1, 10, 5, 5]              20
             ReLU-24             [-1, 10, 5, 5]               0
           Conv2d-25             [-1, 16, 3, 3]           1,440
        AvgPool2d-26             [-1, 16, 1, 1]               0
           Conv2d-27             [-1, 10, 1, 1]             160
================================================================
Total params: 7,932
Trainable params: 7,932
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.51
Params size (MB): 0.03
Estimated Total Size (MB): 0.54
----------------------------------------------------------------
Epoch 1
Train Loss=0.0081 Accuracy=91.78: 100%|██████████| 1875/1875 [01:02<00:00, 30.06it/s]

Test set: Average loss: 0.0838, Accuracy: 9740/10000 (97.40%)

Epoch 2
Train Loss=0.0027 Accuracy=97.37: 100%|██████████| 1875/1875 [01:02<00:00, 29.92it/s]

Test set: Average loss: 0.0548, Accuracy: 9826/10000 (98.26%)

Epoch 3
Train Loss=0.0022 Accuracy=97.84: 100%|██████████| 1875/1875 [01:02<00:00, 29.94it/s]

Test set: Average loss: 0.0327, Accuracy: 9896/10000 (98.96%)

Epoch 4
Train Loss=0.0019 Accuracy=98.08: 100%|██████████| 1875/1875 [01:03<00:00, 29.71it/s]

Test set: Average loss: 0.0332, Accuracy: 9894/10000 (98.94%)

Epoch 5
Train Loss=0.0013 Accuracy=98.69: 100%|██████████| 1875/1875 [01:01<00:00, 30.43it/s]

Test set: Average loss: 0.0223, Accuracy: 9928/10000 (99.28%)

Epoch 6
Train Loss=0.0012 Accuracy=98.83: 100%|██████████| 1875/1875 [01:02<00:00, 30.19it/s]

Test set: Average loss: 0.0219, Accuracy: 9935/10000 (99.35%)

Epoch 7
Train Loss=0.0012 Accuracy=98.86: 100%|██████████| 1875/1875 [01:01<00:00, 30.70it/s]

Test set: Average loss: 0.0209, Accuracy: 9934/10000 (99.34%)

Epoch 8
Train Loss=0.0012 Accuracy=98.81: 100%|██████████| 1875/1875 [01:01<00:00, 30.38it/s]

Test set: Average loss: 0.0207, Accuracy: 9937/10000 (99.37%)

Epoch 9
Train Loss=0.0011 Accuracy=98.95: 100%|██████████| 1875/1875 [01:01<00:00, 30.32it/s]

Test set: Average loss: 0.0202, Accuracy: 9940/10000 (99.40%)

Epoch 10
Train Loss=0.0011 Accuracy=98.91: 100%|██████████| 1875/1875 [01:01<00:00, 30.32it/s]

Test set: Average loss: 0.0208, Accuracy: 9939/10000 (99.39%)

Epoch 11
Train Loss=0.0011 Accuracy=98.89: 100%|██████████| 1875/1875 [01:00<00:00, 31.25it/s]

Test set: Average loss: 0.0200, Accuracy: 9939/10000 (99.39%)

Epoch 12
Train Loss=0.0011 Accuracy=98.88: 100%|██████████| 1875/1875 [00:58<00:00, 32.01it/s]

Test set: Average loss: 0.0202, Accuracy: 9940/10000 (99.40%)

Epoch 13
Train Loss=0.0010 Accuracy=98.93: 100%|██████████| 1875/1875 [00:58<00:00, 32.02it/s]

Test set: Average loss: 0.0206, Accuracy: 9944/10000 (99.44%)

Epoch 14
Train Loss=0.0011 Accuracy=98.93: 100%|██████████| 1875/1875 [00:59<00:00, 31.76it/s]

Test set: Average loss: 0.0204, Accuracy: 9938/10000 (99.38%)

Epoch 15
Train Loss=0.0011 Accuracy=98.95: 100%|██████████| 1875/1875 [00:59<00:00, 31.74it/s]

Test set: Average loss: 0.0202, Accuracy: 9938/10000 (99.38%)


