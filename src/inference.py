import os
import models
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset import ImageDataset
from torch.utils.data import DataLoader

# initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#intialize the model
model = models.model(pretrained=False, requires_grad=False).to(device)
# load the model checkpoint
model_fname = os.path.join(os.path.dirname(__file__), r'../outputs/model.pth')
checkpoint = torch.load(model_fname)
# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

label_fname = os.path.join(os.path.dirname(__file__), r'../data/sample/train_toy.csv')
train_csv = pd.read_csv(label_fname)
# genres = train_csv.columns.values[2:]
labels = train_csv['labels'].values
onehot_labels = [x.split(' ') for x in labels]
onehot_labels = list(set([item for sublist in onehot_labels for item in sublist]))
# prepare the test dataset and dataloader
test_data = ImageDataset(
    train_csv, train=False, test=True
)
test_loader = DataLoader(
    test_data, 
    batch_size=1,
    shuffle=False
)

for counter, data in enumerate(test_loader):
    image, target = data['image'].to(device), data['label']
    # get all the index positions where value == 1
    target_indices = [i for i in range(len(target[0])) if target[0][i] == 1]
    # get the predictions by passing the image through the model
    outputs = model(image)
    outputs = torch.sigmoid(outputs)
    outputs = outputs.detach().cpu()
    sorted_indices = np.argsort(outputs[0])
    best = sorted_indices[-3:]
    string_predicted = ''
    string_actual = ''
    for i in range(len(best)):
        string_predicted += f"{onehot_labels[best[i]]}    "
    for i in range(len(target_indices)):
        string_actual += f"{onehot_labels[target_indices[i]]}    "
    image = image.squeeze(0)
    image = image.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"PREDICTED: {string_predicted}\nACTUAL: {string_actual}")
    infer_fname = os.path.join(os.path.dirname(__file__), r'../outputs/inference_%d.jpg'%(counter))
    plt.savefig(infer_fname)
    # plt.savefig(f"../outputs/inference_{counter}.jpg")
    plt.show()