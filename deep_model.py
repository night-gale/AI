import torch
import torch.nn as nn
import torch.nn.functional as F
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

        self.fc3 = nn.Linear(128, 64)

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