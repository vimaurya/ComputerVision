import os
import torch
from PIL import Image
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader
import sys

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class weatherClassificationModelv0(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(in_features=32 * 16 * 16, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=4)
        self.relu = nn.ReLU()

    def forward(self, x: torch.tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class weatherClassificationModelv1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size = 2, padding = 1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(in_features=32*16*16, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=4)
        self.relu = nn.ReLU()

    def forward(self, x : torch.tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32*16*16)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def save_model(modelSave: torch.nn.Module, model_name: str, version: int):
    model = model_name + "v" + str(version) + ".pth"
    print(model)
    model_path = os.path.join('../', 'models', model)
    print(model_path)
    torch.save(modelSave.state_dict(), model_path)


def predict_weather(model: torch.nn.Module, img_class, img_num):
    class_idx = {0: 'cloudy', 1: 'rainy', 2: 'shiny', 3: 'sunrise'}

    img_path = os.path.join("C:\\large_files\\data_weather\\val", img_class, img_class + str(img_num) + ".jpg")

    if not os.path.exists(img_path):
        img_path = os.path.join("C:\\large_files\\data_weather\\val", img_class, img_class + str(img_num) + ".jpeg")

    if not os.path.exists(img_path):
        print("This image does not exist")
        sys.exit(1)

    print(img_path)

    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        raise IOError(f"Error opening image: {e}")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    img = transform(img).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        pred = model(img)

        return class_idx[pred.argmax(dim=1).item()]


def accuracy_function(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum()
    acc = (correct / len(y_pred)) * 100
    return acc


def train(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer, accuracy_function, device: torch.device = device):
    model.to(device)
    train_loss = 0
    acc = 0
    model.train()
    for batch, (img, label) in enumerate(data_loader):

        img, label = img.to(device), label.to(device)
        train_preds = model(img)
        loss = loss_fn(train_preds, label)
        train_loss += loss
        acc += accuracy_function(y_true=label, y_pred=train_preds.argmax(dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 53 == 0:
            print(f"Looked at {(batch + 1) * len(img)}/{len(data_loader.dataset)} samples.")

    train_loss /= len(data_loader)
    acc /= len(data_loader)

    print(f"Train loss : {train_loss:.4f}, train acc : {acc:2f}%")


def test(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, accuracy_function,
         device: torch.device = device):
    model.to(device)
    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            test_preds = model(x)
            test_loss += loss_fn(test_preds, y)
            test_acc += accuracy_function(y_true=y, y_pred=test_preds.argmax(dim=1))

        test_loss /= len(data_loader)
        test_acc /= len(data_loader)

        print(f"Test loss : {test_loss:.3f}, test acc : {test_acc:.2f}%")
