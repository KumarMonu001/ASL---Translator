import torch
import torch.nn as nn

class ASLCNN(nn.Module):

    def __init__(self, num_classes=29):

        super(ASLCNN, self).__init__()

        # feature extraction

        self.features = nn.Sequential(

            # block 1
            # Input:
            # [batch, 3, 64, 64]


            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            # Output:
            # [BATCH, 32, 32, 32]


            # block 2

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            # Output:
            # [BATCH, 64, 16, 16]

            # block 3

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )

            # Output:
            # [BATCH, 128, 8, 8]
        )


        # classifier

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128 * 8 * 8,
                512
            ),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(
                512,
                num_classes
            )
        )


    # forward pass

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x

# model test
if __name__ == "__main__":

    model = ASLCNN()

    dummy_input = torch.randn(32, 3, 64, 64)

    output = model(dummy_input)

    print("\nOutput Shape:")
    print(output.shape)