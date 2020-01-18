import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeNet2D(nn.Module):
    def __init__(self, input_size, input_channels=3, kernel_sizes=(3, 3, 3), filters=(16, 16, 16),
                 embedding_size=128, classifier=False, n_classes=3, weights_path=None):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, filters[0], kernel_sizes[0], stride=1, padding=1)
        self.conv2 = nn.Conv2d(filters[0], filters[1], kernel_sizes[1], stride=1, padding=1)
        self.conv3 = nn.Conv2d(filters[1], filters[2], kernel_sizes[2], stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)
        self._classifier = classifier
        linear_in = (input_size//8)**2*filters[2]
        if self._classifier:
            self.linear = nn.Linear(linear_in, n_classes)
        else:
            self.linear = nn.Linear(linear_in, embedding_size)

        if weights_path is not None:
            state_dict = torch.load(weights_path)
            try:
                self.load_state_dict(state_dict)
            except RuntimeError:
                state_dict.pop('linear.weight')
                state_dict.pop('linear.bias')
                current_state_dict = self.state_dict()
                current_state_dict.update(state_dict)
                self.load_state_dict(current_state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        x = self.pool(x)
        x = self.conv3(x)
        x = F.relu(x, inplace=True)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        if not self._classifier:
            x = F.normalize(x, p=2, dim=1)
        return x

    def inference(self, x):
        x = self.forward(x)
        if self._classifier:
            x = F.softmax(x, dim=1)
        else:
            x = F.normalize(x, p=2, dim=1)
        return x

    def classifier(self, classifier=True):
        self._classifier = classifier


class NodeNet3D(nn.Module):
    def __init__(self, input_size, input_channels=3, embedding_size=128, classifier=False,
                 n_classes=3, weights_path=None):
        super().__init__()
        self._classifier = classifier

        self.conv = nn.Sequential(
            # Block 1

            nn.Conv3d(input_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, stride=2),

            # Block 2
            nn.Conv3d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, stride=2),

            # Block 3
            nn.Conv3d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, stride=2),
        )
        in_features = 256*(input_size//8)**3
        self.linear = nn.Linear(in_features, embedding_size)
        self.linear_out = nn.Linear(embedding_size, n_classes)

        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path))

    def forward(self, x):
        x = self.conv(x)
        x = x.view((x.size(0), -1))
        x = self.linear(x)
        if self._classifier:
            x = self.linear_out(x)
        else:
            x = F.normalize(x, p=2, dim=1)
        return x

    def inference(self, x):
        x = self.forward(x)
        if self._classifier:
            x = F.softmax(x, dim=1)
        return x

    def classifier(self, classifier=True):
        self._classifier = classifier


# class NodeNet3D(nn.Module):
#     def __init__(self, input_size, input_channels=3, kernel_sizes=(3, 3, 3), filters=(16, 16, 16),
#                  embedding_size=128, classifier=False, n_classes=3, weights_path=None):
#         super().__init__()
#         self.conv1 = nn.Conv3d(input_channels, filters[0], kernel_sizes[0], stride=1, padding=1)
#         self.conv2 = nn.Conv3d(filters[0], filters[1], kernel_sizes[1], stride=1, padding=1)
#         self.conv3 = nn.Conv3d(filters[1], filters[2], kernel_sizes[2], stride=1, padding=1)
#         self.pool = nn.MaxPool3d(2, stride=2)
#         self._classifier = classifier
#         linear_in = (input_size//8)**3*filters[2]
#         if self._classifier:
#             self.linear = nn.Linear(linear_in, n_classes)
#         else:
#             self.linear = nn.Linear(linear_in, embedding_size)
#
#         if weights_path is not None:
#             self.load_weights(weights_path)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x, inplace=True)
#         x = self.pool(x)
#         x = self.conv2(x)
#         x = F.relu(x, inplace=True)
#         x = self.pool(x)
#         x = self.conv3(x)
#         x = F.relu(x, inplace=True)
#         x = self.pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.linear(x)
#         if not self._classifier:
#             x = F.normalize(x, p=2, dim=1)
#         return x
#
#     def inference(self, x):
#         x = self.forward(x)
#         if self._classifier:
#             x = F.softmax(x, dim=1)
#         else:
#             x = F.normalize(x, p=2, dim=1)
#         return x
#
#     def classifier(self, classifier=True):
#         self._classifier = classifier
#
#     def load_weights(self, path):
#         state_dict = torch.load(path)
#         try:
#             self.load_state_dict(state_dict)
#         except RuntimeError:
#             state_dict.pop('linear.weight')
#             state_dict.pop('linear.bias')
#             current_state_dict = self.state_dict()
#             current_state_dict.update(state_dict)
#             self.load_state_dict(current_state_dict)


if __name__ == '__main__':
    import numpy as np

    # 2D network
    x = torch.from_numpy(np.random.random((10, 3, 32, 32))).float()
    net = NodeNet2D(32)
    out1 = net.inference(x)
    print(torch.norm(out1[0], dim=0))  # should be 1
    net.classifier()
    out2 = net.inference(x)
    print(out2[0].sum())  # should be 1

    # 3D network
    x = torch.from_numpy(np.random.random((10, 3, 80, 80, 80))).float()
    net = NodeNet3D(80)
    out1 = net.inference(x)
    print(torch.norm(out1[0], dim=0))  # should be 1
    net.classifier()
    out2 = net.inference(x)
    print(out2[0].sum())  # should be 1
