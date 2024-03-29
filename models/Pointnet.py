# code from https://github.com/yanx27/Pointnet_Pointnet2_pytorch
import numpy as np
import torch
from torchsummary import summary


class TransformationNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim, device='cuda:0'):
        super(TransformationNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv1 = torch.nn.Conv1d(self.input_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, input_dim * output_dim)
        self.relu = torch.nn.ReLU()

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.bn5 = torch.nn.BatchNorm1d(256)
        self.device = device

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        identity_matrix = torch.eye(self.output_dim)
        if torch.cuda.is_available():
            identity_matrix = identity_matrix.to(self.device)
        x = x.view(-1, self.output_dim, self.output_dim) + identity_matrix
        return x

class PointNetEncoder(torch.nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3, device='cuda:0'):
        super(PointNetEncoder, self).__init__()
        self.stn = TransformationNet(channel, channel, device=device)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.relu = torch.nn.ReLU()
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = TransformationNet(64, 64, device=device)
        self.device = device

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)     
        x = x.transpose(2, 1)       # B, N, D
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)       # B, D, N
        x = self.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)       # B, N, D
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)       # B, D, N
        else:
            trans_feat = None

        pointfeat = x
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)            # global feature
        if self.global_feat:
            return x
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNetClass(torch.nn.Module):
    def __init__(self, k=10, normal_channel=False):
        super(PointNetClass, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, k)
        self.dropout = torch.nn.Dropout(p=0.4)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.relu = torch.nn.ReLU()
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = x.transpose(2, 1)       # from (B, N, D) to (B, D, N)
        x = self.feat(x)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = self.log_softmax(x)
        return x

class PointNetSeg(torch.nn.Module):
    def __init__(self, num_class, device='cuda:0'):
        super(PointNetSeg, self).__init__()
        self.k = num_class
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=3, device=device)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.bn3 = torch.nn.BatchNorm1d(128)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.relu = torch.nn.ReLU()
        #self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = x.transpose(2, 1)       # from (B, N, D) to (B, D, N)
        batchsize = x.size(0)
        n_pts = x.size(2)
        x, trans, trans_feat = self.feat(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.dropout(self.conv2(x))))
        x = self.relu(self.bn3(self.dropout(self.conv3(x))))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()           # (B, N, C)
        x = x.view(batchsize, n_pts)
        return x, trans_feat

def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.momentum = momentum

def feature_transform_reguliarzer(trans, device='cuda:0'):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.to(device)
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

if __name__ == '__main__':
    #x_rand = torch.Tensor(np.random.random((32, 1024, 3)) * 300).to(device=device)
    device = 'cuda:0'
    model = PointNetSeg(1).to(device)
    summary(model, (2048, 3))
    model = PointNetSeg(1, device).to(device)
    for i in range(100):
        x_rand = torch.Tensor(np.random.random((32, 1024+i, 3)) * 300).to(device=device)
        yout, _= model(x_rand)
        print(yout.size())
