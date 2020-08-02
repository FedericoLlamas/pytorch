import torch
import torchvision
import torch.nn as nn
from models import Net
from utils import plot
import torch.optim as optim
import torch.nn.functional as F
import os


def test(network, datasets):
    test_losses = []
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in datasets['val']:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(datasets['val'].dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(datasets['val'].dataset),
        100. * correct / len(datasets['val'].dataset)))


def train(network, optimizer, datasets, epoch):
    train_losses = []
    train_counter = []
    log_interval = 10
    network.train()
    for batch_idx, (data, target) in enumerate(datasets['train']):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(datasets['train'].dataset),
                100. * batch_idx / len(datasets['train']), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(datasets['train'].dataset)))
            torch.save(network.state_dict(), './model.pth')
            torch.save(optimizer.state_dict(), './optimizer.pth')


def run():
    learning_rate = 0.01
    momentum = 0.5
    batch_size_train = 64
    batch_size_test = 1000
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # ------------------------------------------------------------------------

    datasets = {
        "train": torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                './mnist_dataset', train=True, download=True,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
            batch_size=batch_size_train, shuffle=True
        ),
        "val": torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                './mnist_dataset', train=False, download=True,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
            batch_size=batch_size_test, shuffle=True)
    }

    examples = enumerate(datasets['val'])
    batch_idx, (example_data, example_targets) = next(examples)

    # ------------------------------------------------------------------------

    network = Net()

    optimizer = optim.SGD(
        network.parameters(), lr=learning_rate, momentum=momentum)

    # ------------------------------------------------------------------------

    n_epochs = 3

    """
        Muestro una clasificación con el modelo sin aprender (random)
        Luego testeo el Accuracy obtenido
    """
    with torch.no_grad():
        output = network(example_data)
    plot(example_data, output)
    test(network, datasets)

    if os.path.exists('model.pth'):
        # Cargo el modelo aprendido
        network.load_state_dict(torch.load('./model.pth'))
    else:
        # Entreno el modelo y luego testeo el Accuracy obtenido
        for epoch in range(1, n_epochs + 1):
            train(network, optimizer, datasets, epoch)
            test(network, datasets)
    """
    Muestro la clasificación predicha con el modelo aprendido y
    testeo el accuracy
    """
    with torch.no_grad():
        output = network(example_data)
    plot(example_data, output)
    test(network, datasets)


if __name__ == "__main__":
    run()
