import argparse
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

use_cuda = torch.cuda.is_available()
device = "cuda"


def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer, checkpoint["epoch"]


def save_ckp(state, is_best, checkpoint_dir):
    """
     Save the state to the specified checkpoint directory. In addition to that, if the model is the best model till now,
     the same checkpoint is also copied to another directory to keep track of the best model.
    :param state:
    :param is_best:
    :param checkpoint_dir:
    :param best_model_dir:
    :return:
    """
    f_path = checkpoint_dir / "checkpoint.pt"
    torch.save(state, f_path)
    if is_best:
        print("New best model!")
        best_fpath = checkpoint_dir / "best_model.pt"
        shutil.copyfile(f_path, best_fpath)


def prepare_dataloaders(data_dir, batch):
    # Loading the MNIST Dataset from Pytorch
    # Import MNIST dataset
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=True, transform=transforms.ToTensor(), download=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=False, transform=transforms.ToTensor()
    )

    # Importing the Dataloader & specify Batch size
    # Data loader
    train_loader_out = DataLoader(dataset=train_dataset, batch_size=batch, shuffle=True)

    test_loader_out = DataLoader(dataset=test_dataset, batch_size=batch, shuffle=False)

    return train_loader_out, test_loader_out


class NeuralNet(nn.Module):
    # Fully connected neural network with one hidden layer
    def __init__(self, size_in, size_hidden, n_classes):
        super(NeuralNet, self).__init__()
        self.input_size = size_in
        self.l1 = nn.Linear(size_in, size_hidden)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(size_hidden, n_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out


if __name__ == '__main__':
    print(f"PyTorch {torch.__version__}")
    print(f"On GPU: {use_cuda}")

    parser = argparse.ArgumentParser(description='Checkpointing PyTorch Models.')
    # '/work/bootcamp/tutorials/'
    parser.add_argument('-d', '--data', type=str, default='../data/',
                        help='Directory of MNIST: if MNIST is in "data," load; else, download.')
    parser.add_argument('-e', '--epochs', type=int, default=2,
                        help='The number of epochs to train model.')
    parser.add_argument('-v', '--val', type=int, default=1,
                        help='Run validation every v-th epoch.')
    parser.add_argument('-o', '--out', type=str, default='training_4',
                        help='Directory to save checkpoints.')


    args = parser.parse_args()
    # hyperparameters for our neural network
    num_epochs = args.epochs
    input_size = 784  # 28x28
    hidden_size = 500
    num_classes = 10
    # num_epochs = 2
    batch_size = 100
    learning_rate = 0.001

    train_loader, test_loader = prepare_dataloaders(args.data, batch_size)
    # epoch number of steps for each job, get it as a commandline argument:

    # Include the epoch in the file name (uses `str.format`)
    # checkpoint_path = "training_4/{epoch:d}.ckpt"
    model_dir = args.out
    # checkpoint_dir = f"{model_dir}/checkpoint/"
    model_path = Path(model_dir)
    model_path.mkdir(exist_ok=True, parents=True)

    # Create a new model instance
    model = NeuralNet(input_size, hidden_size, num_classes)
    model = model.to("gpu") if use_cuda else model

    # Setting our Loss and Optimizer Functions
    # We are using Adam optimizer here.

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    n_total_steps = len(train_loader)
    top_accuracy = 0.0
    for epoch in range(num_epochs):
        for nbatch, (images, labels) in enumerate(train_loader):
            # origin shape: [100, 1, 28, 28]
            # resized: [100, 784]
            images = images.reshape(-1, input_size)
            if use_cuda:
                images = images.to(device)
                labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # To Print the Loss at every 100th step and show our total steps
            if (nbatch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step[{nbatch + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

        if (epoch + 1) % args.val == 0 or epoch == num_epochs - 1:
            # every v-th epoch AND the last epoch
            print("Testing of the Model and Evaluating Accuracy")
            with torch.no_grad():
                n_correct = 0
                n_samples = 0
                for images, labels in test_loader:
                    images = images.reshape(-1, input_size)

                    if use_cuda:
                        images = images.to(device)
                        labels = labels.to(device)

                    # images = images.reshape(-1, 28 * 28).to(device)
                    # labels = labels.to(device)
                    outputs = model(images)
                    # max returns (value ,index)
                    _, predicted = torch.max(outputs.data, 1)
                    n_samples += labels.size(0)
                    n_correct += (predicted == labels).sum().item()

                acc = 100.0 * n_correct / n_samples
                # Accuracy of the network on the 10,000 test images: 97.3%
                print(f'Accuracy of the network on epoch {epoch + 1}: {acc} %')

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': acc,
                }
                save_ckp(checkpoint, acc > top_accuracy, model_path)

