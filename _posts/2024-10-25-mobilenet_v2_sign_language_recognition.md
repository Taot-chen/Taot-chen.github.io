---
layout: post
title: mobilenet_v2_sign_language_recognition
date: 2024-10-25
tags: [deeplearning]
author: taot
---

## 基于mobilenet_v2自训练手语识别模型

github 项目地址：https://github.com/Taot-chen/raspberrypi_dl

### 1 数据集

手语识别数据集，每个字母有一种手势表达，共26个类别。
* 训练集：26个字母每个字母约130张图片
* 测试集：26个字母每个字母约100张图片

数据集：https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet/data

#### 1.1 Linux 通过 kaggle api 下载 kaggle 数据集

在 Linux 环境中通过 kaggle api 下载 kaggle 数据集，可以避免将数据集下载到本地，再上传到服务器的繁琐操作，能够直接在 Terminal 操作，在服务器上下载 kaggle 数据集。

* kaggle 登录后下载 API token
  `Your Profile --> Setting --> Create New Token --> Continue`
  会自动将 `kaggle.json` 下载至本地。

* linux 本机安装 kaggle api
  ```bash
  python3 -m pip install kaggle
  ```
* 在`home`目录下创建`.kaggle`文件夹，并把`kaggle.json`放入(Ensure kaggle.json is in the location ~/.kaggle/kaggle.json to use the API.)
  ```bash
  mkdir ~/.kaggle
  ```
* 下载相应数据集
  在对应数据集上找到API命令，使用给出的 API 命令下载即可。
  kaggle 提供了多种下载方式，都可。数据集比较大的时候，通常下载会比较耗时，可以将下载命令放在后台执行，只需要在命令末尾添加 `&` 即可。
  *注：如果下载一段时间之后速度变得很慢了，可以取消掉下载任务，再重新启动任务，可以使速度恢复*

  我这里是使用的 kagglehub 的方式下载的，下载脚本：
  ```python
  import kagglehub

  # Download latest version
  path = kagglehub.dataset_download("lexset/synthetic-asl-alphabet")

  print("Path to dataset files:", path)
  ```

### 2 数据加载

#### 2.1 构建数据集标签文件

`labels.json`
```python
{
  "0": "A",
  
  "1": "B",
 
  "2": "C",
 
  "3": "D",
 
  "4": "E",
 
  "5": "F",
 
  "6": "G",
 
  "7": "H",
 
  "8": "I",
 
  "9": "J",
 
  "10": "K",
 
  "11": "L",
 
  "12": "M",
 
  "13": "N",
 
  "14": "O",
 
  "15": "P",
 
  "16": "Q",
  
  "17": "R",
  
  "18": "S",
  
  "19": "T",
  
  "20": "U",
  
  "21": "V",
  
  "22": "W",
  
  "23": "X",
  
  "24": "Y",
  
  "25": "Z"
 
}
```



#### 2.2 定义一个随机灰度化的概率

```python
random_grayscale_p = 0.25
```
随机灰度化是为了在训练过程中，有一定的概率将图像转换为灰度图，这可以帮助我们的模型更好地泛化。

#### 2.3 定义训练数据加载器所需的变换

```python

transform_train = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.RandomGrayscale(p=random_grayscale_p),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

* Resize：将图像调整到指定的高度和宽度。
* RandomGrayscale：以一定的概率将图像转换为灰度图。
* ToTensor：将图像数据转换为PyTorch的Tensor格式
* Normalize：对图像进行标准化处理，使数据分布在[-1, 1]之间

#### 2.4 定义验证数据加载器

```python
transform_val = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

对于验证数据加载器，不需要随机灰度化。


#### 2.5 利用`torchvision.datasets.ImageFolder`加载图像数据集

```python
train_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform_train)
val_dataset = torchvision.datasets.ImageFolder(root=test_data_dir, transform=transform_val)
```

`torchvision.datasets.ImageFolder`这个类会自动读取指定目录下的图像，并将它们分为不同的类别，但是这个对于数据集的目录结构有要求，我这里的数据集目录结构如下：
```bash
.
├── get_dataset.py
├── infer.py
├── labels.json
├── synthetic-asl-alphabet_dataset
│   ├── alphabet.jpg
│   ├── Test_Alphabet
│   │   ├── A
│   │   ├── B
│   │   ├── Blank
│   │   ├── C
│   │   ├── D
│   │   ├── E
│   │   ├── F
│   │   ├── G
│   │   ├── H
│   │   ├── I
│   │   ├── J
│   │   ├── K
│   │   ├── L
│   │   ├── M
│   │   ├── N
│   │   ├── O
│   │   ├── P
│   │   ├── Q
│   │   ├── R
│   │   ├── S
│   │   ├── T
│   │   ├── U
│   │   ├── V
│   │   ├── W
│   │   ├── X
│   │   ├── Y
│   │   └── Z
│   └── Train_Alphabet
│       ├── A
│       ├── B
│       ├── Blank
│       ├── C
│       ├── D
│       ├── E
│       ├── F
│       ├── G
│       ├── H
│       ├── I
│       ├── J
│       ├── K
│       ├── L
│       ├── M
│       ├── N
│       ├── O
│       ├── P
│       ├── Q
│       ├── R
│       ├── S
│       ├── T
│       ├── U
│       ├── V
│       ├── W
│       ├── X
│       ├── Y
│       └── Z
├── train.py
└── utils.py
```

#### 2.6 利用`torch.utils.data.DataLoader`来创建数据加载器

```python
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
```

`torch.utils.data.DataLoader`这个类会在训练时帮助我们有效地批量加载数据，shuffle=True表示在训练过程中每个epoch都会打乱数据顺序。

#### 2.7 从训练数据集中获取类别名称

```python
class_names = train_dataset.classes
```

#### 2.8 完整代码

`utils.py`
```python
import torch
import torchvision
from torchvision import transforms
 
def data_load(data_dir, test_data_dir, img_height, img_width, batch_size):
    random_grayscale_p = 0.2
    transform_train = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.RandomGrayscale(p=random_grayscale_p),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_val = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform_train)
    val_dataset = torchvision.datasets.ImageFolder(root=test_data_dir, transform=transform_val)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    class_names = train_dataset.classes
    return train_loader, val_loader, class_names
```


### 3 模型训练

#### 3.1 模型加载

```python
def load_model(class_num=27):
    mobilenet = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT)
    mobilenet.classifier[1] = nn.Linear(in_features=mobilenet.classifier[1].in_features, out_features=class_num)
    return mobilenet
```

模型使用预训练的 mobilenet_v2 模型，并且修改模型的最后一层（分类器），使其输出的类别数为 class_num。


#### 3.2 定义训练函数 train

```python
def train(epochs):
    train_loader, val_loader, class_names = utils.data_load("./synthetic-asl-alphabet_dataset/Train_Alphabet", "./synthetic-asl-alphabet_dataset/Test_Alphabet", 224, 224, 256)
    print("类别名称：", class_names)
    model = load_model(class_num=len(class_names))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if os.path.exists("mobilenet_latest.pth"):
        model.load_state_dict(torch.load("mobilenet_latest.pth"))
        print("加载已有模型继续训练")
 
    best_val_accuracy = 0.0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    patience = 5
    patience_counter = 0
 
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(correct / total)
        print("训练损失：{:.4f}，准确率：{:.4f}".format(train_losses[-1], train_accuracies[-1]))
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_losses.append(running_loss / len(val_loader))
        val_accuracies.append(correct / total)
        print("验证损失：{:.4f}，准确率：{:.4f}".format(val_losses[-1], val_accuracies[-1]))
        if val_accuracies[-1] > best_val_accuracy:
            best_val_accuracy = val_accuracies[-1]
            torch.save(model.state_dict(), "mobilenet_latest.pth")
            print("模型已保存")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("早停触发，停止训练")
                break
```
训练函数就很常规，没什么好说的。

#### 3.3 完整代码

`train.py`
```python
import os
import torch
import torch.nn as nn
import torchvision
import utils
 
 
def load_model(class_num=27):
    mobilenet = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT)
    mobilenet.classifier[1] = nn.Linear(in_features=mobilenet.classifier[1].in_features, out_features=class_num)
    return mobilenet
 
def train(epochs):
    train_loader, val_loader, class_names = utils.data_load("./synthetic-asl-alphabet_dataset/Train_Alphabet", "./synthetic-asl-alphabet_dataset/Test_Alphabet", 224, 224, 256)
    print("类别名称：", class_names)
    model = load_model(class_num=len(class_names))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if os.path.exists("mobilenet_latest.pth"):
        model.load_state_dict(torch.load("mobilenet_latest.pth"))
        print("加载已有模型继续训练")
 
    best_val_accuracy = 0.0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    patience = 5
    patience_counter = 0
 
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(correct / total)
        print("训练损失：{:.4f}，准确率：{:.4f}".format(train_losses[-1], train_accuracies[-1]))
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_losses.append(running_loss / len(val_loader))
        val_accuracies.append(correct / total)
        print("验证损失：{:.4f}，准确率：{:.4f}".format(val_losses[-1], val_accuracies[-1]))
        if val_accuracies[-1] > best_val_accuracy:
            best_val_accuracy = val_accuracies[-1]
            torch.save(model.state_dict(), "mobilenet_latest.pth")
            print("模型已保存")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("早停触发，停止训练")
                break
 
if __name__ == '__main__':
    train(epochs=50)
```

### 4 模型推理

#### 4.1 定义 SignLanguageRecognition 类初始化函数 __init__

```python
def __init__(self, module_file='./mobilenet_latest.pth', labels_file='./labels.json'):
    self.module_file = module_file
    self.CUDA = torch.cuda.is_available()
    self.net = mobilenet_v2(num_classes=27)
    if self.CUDA:
        self.net.cuda()
    self.net.load_state_dict(torch.load(self.module_file, map_location='cuda' if self.CUDA else 'cpu'))
    self.net.eval()
    self.labels = self.load_labels(labels_file)
```

#### 4.2 加载标签

```python
def load_labels(self, labels_path):
    with open(labels_path, 'r', encoding='utf-8') as file:
        return json.load(file)
```

#### 4.3 图像预处理函数

```python
@torch.no_grad()
def preprocess_image(self, image_stream):
    img = Image.open(image_stream)
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.56719673, 0.5293289, 0.48351972], std=[0.20874391, 0.21455203, 0.22451781]),
    ])
    img = transform(img)
    img = torch.unsqueeze(img, 0)
    if self.CUDA:
        img = img.cuda()
    return img
```

* 调整图像大小到224x224像素。
* 将图像转换为张量。
* 对张量进行归一化。
* 增加一个维度，以便适合模型输入。
* 将图像张量移动到推理设备上。

#### 4.4 图像识别

```python
def predict(self, image_stream):
    probs, cls = self.recognize(image_stream)
    _, cls = torch.max(probs, 1)
    p = probs[0][cls.item()]
    cls_index = str(cls.numpy()[0])
    label_name = self.labels.get(cls_index, "未知标签")
    return label_name, p.item()
```

调用 recognize 函数获取预测结果，然后从标签字典中查找对应的标签名称，并返回标签名称和最大概率。

#### 4.5 创建实例并进行预测

```python
recongize=SignLanguageRecognition()
ret = recongize.predict('./synthetic-asl-alphabet_dataset/Test_Alphabet/A/e4761e88-10df-41d2-b980-463375c5c46c.rgb_0000.png')
print(ret)
ret = recongize.predict('./synthetic-asl-alphabet_dataset/Test_Alphabet/Blank/5da53b24-d860-4921-a645-9fe85bd91213.rgb_0000.png')
print(ret)
```

#### 4.6 完整代码

`infer.py`

```python
import json
from torchvision.models import mobilenet_v2
import torch
import torch.nn.functional as F
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from PIL import Image
 
class SignLanguageRecognition:
    def __init__(self, module_file='./mobilenet_latest.pth', labels_file='./labels.json'):
        self.module_file = module_file
        self.CUDA = torch.cuda.is_available()
        self.net = mobilenet_v2(num_classes=27)
        if self.CUDA:
            self.net.cuda()
        self.net.load_state_dict(torch.load(self.module_file, map_location='cuda' if self.CUDA else 'cpu'))
        self.net.eval()
        self.labels = self.load_labels(labels_file)

    def load_labels(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
 
    @torch.no_grad()
    def preprocess_image(self, image_stream):
        img = Image.open(image_stream)
        transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.56719673, 0.5293289, 0.48351972], std=[0.20874391, 0.21455203, 0.22451781]),
        ])
        img = transform(img)
        img = torch.unsqueeze(img, 0)
        if self.CUDA:
            img = img.cuda()
        return img

    @torch.no_grad()
    def recognize(self, image_stream):
        img = self.preprocess_image(image_stream)
        y = self.net(img)
        y = F.softmax(y, dim=1)
        p, cls_idx = torch.max(y, dim=1)
        return y.cpu(), cls_idx.cpu()
 
    def predict(self, image_stream):
        probs, cls = self.recognize(image_stream)
        _, cls = torch.max(probs, 1)
        p = probs[0][cls.item()]
        cls_index = str(cls.numpy()[0])
        label_name = self.labels.get(cls_index, "未知标签")
        return label_name, p.item()

if __name__ == "__main__": 
    recongize=SignLanguageRecognition()
    ret = recongize.predict('./synthetic-asl-alphabet_dataset/Test_Alphabet/A/e4761e88-10df-41d2-b980-463375c5c46c.rgb_0000.png')
    print(ret)
    ret = recongize.predict('./synthetic-asl-alphabet_dataset/Test_Alphabet/Blank/5da53b24-d860-4921-a645-9fe85bd91213.rgb_0000.png')
    print(ret)
```

推理示例：
```python
python infer.py
('A', 0.9999058246612549)
('Blank', 0.9999775886535645)
```
