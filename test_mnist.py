import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from remixier import RemixierImageClassificator
from mnist import testset, testloader, trainset, trainloader

model = RemixierImageClassificator(
    channels=1,
    image_size=28,
    patch_size=7,
    classes=10,
    dim=128,
    num_layers=4
)

epochs = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
model.to(device)

def main():
    if not os.path.exists("model.pt"):
        bar = tqdm(total=(epochs*len(trainset)))
        for epoch in range(epochs):
            for i, (image, label) in enumerate(trainloader):
                image, label = image.to(device), label.to(device)
                optimizer.zero_grad()
                output = model(image)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

                bar.set_description(f"Epoch: {epoch} Loss: {loss.item()}")
                bar.update(image.shape[0])
        torch.save(model, "model.pt")
    #test
    correct = 0
    total = 0
    with torch.no_grad():
        for image, label in testloader:
            image, label = image.to(device), label.to(device)
            output = model(image)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    print(f'Accuracy of the network on the test images: {(100 * correct / total)}')


    
if __name__ == "__main__":
    main()