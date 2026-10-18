import torch
import torch.nn as nn


class ZeroDCE(nn.Module):

    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        self.e_conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.e_conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.e_conv3 = nn.Conv2d(32, 32, 3, 1, 1)
        self.e_conv4 = nn.Conv2d(32, 32, 3, 1, 1)

        self.e_conv5 = nn.Conv2d(64, 32, 3, 1, 1)
        self.e_conv6 = nn.Conv2d(64, 32, 3, 1, 1)
        self.e_conv7 = nn.Conv2d(64, 24, 3, 1, 1)

    def forward(self, x):

        x1 = self.relu(self.e_conv1(x))

        x2 = self.relu(self.e_conv2(x1))

        x3 = self.relu(self.e_conv3(x2))

        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(
            self.e_conv5(
                torch.cat([x3, x4], dim=1)
            )
        )

        x6 = self.relu(
            self.e_conv6(
                torch.cat([x2, x5], dim=1)
            )
        )

        x_r = torch.tanh(
            self.e_conv7(
                torch.cat([x1, x6], dim=1)
            )
        )

        return x_r


def enhance_image(x, model):

    curve = model(x)

    enhanced = x

    for _ in range(8):

        enhanced = enhanced + curve * (
            enhanced * enhanced - enhanced
        )

    return torch.clamp(enhanced, 0, 1)