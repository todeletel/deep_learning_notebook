import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 784 * 2)
        self.fc2 = nn.Linear(784 * 2, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 10)
    
    def forward(self, x):
        input_vector = x.view(-1,784)
        fc1_out = F.relu(self.fc1(input_vector))
        fc2_out = F.relu(self.fc2(fc1_out))
        fc3_out = F.relu(self.fc3(fc2_out))
        output = self.fc4(fc3_out)
        return output

model = Net()

criterion = nn.CrossEntropyLoss()
optim = torch.optim = torch.optim.Adam(model.parameters(),lr=0.001)

model.cpu()

total = 0
correct = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Before Trianing Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))

model.cuda()

epoch = 10
for i in range(epoch):
    run_loss = 0
    j = 0
    for img,label in trainloader:
        img_gpu,label_gpu = img.to('cuda'), label.to('cuda')
        output = model(img_gpu)
        optim.zero_grad()
        loss = criterion(output, label_gpu)
        loss.backward()
        optim.step()
        run_loss += loss
        j += 1
        if j % 200 == 0:
            print(f"epoch is {i} iter is {j},loss is {run_loss/j}")

model.cpu()
total = 0
correct = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('After Trianing Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))