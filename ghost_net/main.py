import vgg
import ghost_vgg
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from utils import progress_bar
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True
)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True
)


print('Model')
# net = vgg.VGG('VGG16')
net = ghost_vgg.VGG_GHOST('VGG16')
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.bechmark = True

# assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
# checkpoint = torch.load('./checkpoint/ckpt.pth')
# net.load_state_dict(checkpoint['net'])
# best_acc = checkpoint['acc']
# start_epoch = checkpoint['epoch']

print(f'classes: {trainset.classes}')
classes = trainset.classes

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

train_accLst = []
train_lossLst = []

def train(epoch):
    print(f'\nEpoch: {epoch}')
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100.*correct/total
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                %(train_loss/(batch_idx+1), acc, correct, total))
    train_accLst.append(acc)
    train_lossLst.append(train_loss/(batch_idx+1))

test_accLst = []
test_lossLst = []
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' %(test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
    acc = 100.*correct/total
    test_lossLst.append(loss)
    test_accLst.append(acc)
    if acc > best_acc:
        state = {
            'net':net.state_dict(),
            'acc':acc,
            'epoch':epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
   
for epoch in range(start_epoch, start_epoch+100):
    train(epoch)
    test(epoch)
    scheduler.step()


# test_accLst = test_accLst.detach().cpu().numpy()
# test_lossLst = test_lossLst.detach().cpu().numpy()
# train_accLst = train_accLst.detach().cpu().numpy()
# train_lossLst = train_lossLst.detach().cpu().numpy()
plt.plot(range(0, 100), test_accLst, label='test')
plt.plot(range(0, 100), train_accLst, label='train')
plt.legend()
plt.title('Accuracy[VGG16]')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
plt.plot(range(0, 100), test_lossLst, label='test')
plt.plot(range(0, 100), train_lossLst, label='train')
plt.legend()
plt.title('Loss[VGG16]')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

