import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from numpy.lib.stride_tricks import as_strided
# deep model
class OthelloNNet(nn.Module):
    def __init__(self, ):
        # game params
        # hard coded
        self.board_x, self.board_y = (8, 8)

        super(OthelloNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 128,  3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, stride=1)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * (self.board_x - 4) * (self.board_y - 4), 128)
        self.fc_bn1 = nn.BatchNorm1d(128)
        
        self.fc2 = nn.Linear(128, 128)
        self.fc_bn2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, 65)

        self.fc4 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = x.view(-1, 1, self.board_x, self.board_y)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = x.view(-1, 128 * 4 * 4)

        x = F.relu(self.fc_bn1(self.fc1(x)))
        x = F.relu(self.fc_bn2(self.fc2(x)))

        pi = self.fc3(x)

        v = self.fc4(x)

        return F.log_softmax(pi, dim=1), torch.tanh(v)

    def predict(self, x):
        x = torch.FloatTensor(x.astype(np.float64))

        self.eval()

        with torch.no_grad():
            pi, v = self.forward(x)
        
        return torch.exp(pi).data.numpy()[0], v.data.numpy()[0]

    def load(self, path):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.load_state_dict(checkpoint['state_dict'])

class NeuralNet:
    def __init__(self) -> None:
        self.conv1 = {}
        self.conv_bn1 = {}
        self.conv2 = {}
        self.conv_bn2 = {}
        self.conv3 = {}
        self.conv_bn3 = {}
        self.conv4 = {}
        self.conv_bn4 = {}

        self.fc1 = {}
        self.fc_bn1 = {}
        self.fc2 = {}
        self.fc_bn2 = {}
        self.fc3 = {}
        self.fc4 = {}

    def forward(self, x):
        x = conv2d(x, self.conv1['weight'], self.conv1['bias'], padding=1)
        x = bn(x, self.conv_bn1['weight'], self.conv_bn1['bias'], self.conv_bn1['running_mean'], self.conv_bn1['running_var'])
        x = relu(x)
        x = conv2d(x, self.conv2['weight'], self.conv2['bias'], padding=1)
        x = bn(x, self.conv_bn2['weight'], self.conv_bn2['bias'], self.conv_bn2['running_mean'], self.conv_bn2['running_var'])
        x = relu(x)
        x = conv2d(x, self.conv3['weight'], self.conv3['bias'], )
        x = bn(x, self.conv_bn3['weight'], self.conv_bn3['bias'], self.conv_bn3['running_mean'], self.conv_bn3['running_var'])
        x = relu(x)
        x = conv2d(x, self.conv4['weight'], self.conv4['bias'], )
        x = bn(x, self.conv_bn4['weight'], self.conv_bn4['bias'], self.conv_bn4['running_mean'], self.conv_bn4['running_var'])
        x = relu(x)

        x = x.transpose([0, 3, 1, 2])
        x = x.reshape([1, -1])
        x = fc(x, self.fc1['weight'], self.fc1['bias'])
        x = bn(x, self.fc_bn1['weight'], self.fc_bn1['bias'], self.fc_bn1['running_mean'], self.fc_bn1['running_var'])
        x = relu(x)
        x = fc(x, self.fc2['weight'], self.fc2['bias'])
        x = bn(x, self.fc_bn2['weight'], self.fc_bn2['bias'], self.fc_bn2['running_mean'], self.fc_bn2['running_var'])
        x = relu(x)
        p = fc(x, self.fc3['weight'], self.fc3['bias'])
        val = fc(x, self.fc4['weight'], self.fc4['bias'])

        return softmax(p), np.tanh(val)

    def load_param(self, model:OthelloNNet):
        self.conv1 = toNumpy(model.conv1.state_dict())
        self.conv1['weight'] = self.conv1['weight'].transpose([2, 3, 1, 0])
        self.conv2 = toNumpy(model.conv2.state_dict())
        self.conv2['weight'] = self.conv2['weight'].transpose([2, 3, 1, 0])
        self.conv3 = toNumpy(model.conv3.state_dict())
        self.conv3['weight'] = self.conv3['weight'].transpose([2, 3, 1, 0])
        self.conv4 = toNumpy(model.conv4.state_dict())
        self.conv4['weight'] = self.conv4['weight'].transpose([2, 3, 1, 0])

        self.conv_bn1 = toNumpy(model.bn1.state_dict())
        self.conv_bn2 = toNumpy(model.bn2.state_dict())
        self.conv_bn3 = toNumpy(model.bn3.state_dict())
        self.conv_bn4 = toNumpy(model.bn4.state_dict())

        self.fc1 = toNumpy(model.fc1.state_dict())
        self.fc2 = toNumpy(model.fc2.state_dict())
        self.fc3 = toNumpy(model.fc3.state_dict())
        self.fc4 = toNumpy(model.fc4.state_dict())

        self.fc_bn1 = toNumpy(model.fc_bn1.state_dict())
        self.fc_bn2 = toNumpy(model.fc_bn2.state_dict())
    
def toNumpy(state_dict):
    for key in state_dict:
        state_dict[key] = state_dict[key].data.numpy()
    return state_dict
        

# TODO
# Stride & padding
def conv2d(x, w, b, padding=False):
    """
    param x: ndarray of shape (N, H, W, Cin), input tensor \n
    param w: ndarray of shape (K, K, Cin, Cout), the weights of kxk kernel \n
    param b: ndarray of shape (Cout), the bias term \n
    return ndarray of shape (N, H - 2, W - 2, Cout) 
    """
    N, H, W, Cin = x.shape
    if padding:
        pad = np.zeros((N, H + 2, W + 2, Cin))
        pad[:, 1:H+1, 1:W+1, :] = x
        x = pad
        H = H + 2
        W = W + 2
    K = w.shape[0]
    Hout = x.shape[1] - K + 1
    Wout = x.shape[2] - K + 1
    x = as_strided(x, (x.shape[0], Hout, Wout, w.shape[0], w.shape[1], x.shape[3]), x.strides[:3] + x.strides[1:])
    # np.repeat(x, w.shape[3], axis=-1)
    return np.tensordot(x, w, axes=3) + b
    
def fc(x, w, b):
    """
    param x: ndarray of shape (N, H) \n
    param w: ndarray of shape (M, H) \n
    param b: ndarray of shape (M, )
    """
    return x.dot(w.T) + b
    
def bn(x, w, b, mean, var, eps=1e-5):
    """
    param x: ndarray of shape (N, H, W, C)/(N, C) \n
    param w: ndarray of shape (C, ) \n
    param b: ndarray of shape (C, )
    """
    return (x - mean) / np.sqrt(var + eps) * w + b
    
def relu(x):
    return np.maximum(x, 0)
    
def saveParam(model: OthelloNNet):
    for module in model.modules():
        for param in module.parameters():
            pass

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def hello():
    print("hello")