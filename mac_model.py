'''
Setting up neural network, code based on the walkthrough from here:
https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5
'''

import torch.nn.functional as F 
from torch.nn import init 
import torch

class AudioClassifier (torch.nn.Module):
    def __init__(self):
        super().__init__()
        conv_layer = []
        
  # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = torch.nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = torch.nn.ReLU()
        self.bn2 = torch.nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Second Convolution Block
        self.conv3 = torch.nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = torch.nn.ReLU()
        self.bn3 = torch.nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Second Convolution Block
        self.conv4 = torch.nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = torch.nn.ReLU()
        self.bn4 = torch.nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear Classifier
        self.ap = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = torch.nn.Linear(in_features=64, out_features=10)

        # Wrap the Convolutional Blocks
        self.conv = torch.nn.Sequential(*conv_layers)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        x = self.lin(x)
        return x

# Define audio file
audioModel = AudioClassifier()
# Define testing device (cpu for me)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
audioModel = audioModel.to(device)
next(audioModel.parameters()).device

# Training functions, defines optimizer, loss, and other functions
def training(model, train_dl, nepochs):
    criteria = torch.nn.CrossEntroypyLoss()
    optimizer = optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=int(len(train_dl)), epochs=nepochs, anneal_strategy='linear')
    
    # Repeat for each epoch
    for epoch in range(nepochs):
        # Loss and preditions
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0
        
        # Repeat for each batch in training set
        for i, data in enumerate(train_dl):
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criteria(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            # Keep stats for accuracy
            running_loss += loss.item()
            
            # Predicted class with highest score
            _, prediction = torch.max(outputs, 1)
            # Count of predictions that matched label instrument
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]
            
            #Print stats at end of epoch
            num_batches = len(train_dl)
            avg_loss = running_loss / num_batches
            acc = correct_prediction / total_prediction
            print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')
            
        print('Finished Testing')
        
        nepochs = 16
        training(audioModel, train_dl, nepochs)
            