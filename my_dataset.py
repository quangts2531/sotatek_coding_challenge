import os
from PIL import Image
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms.v2 import Compose, ToTensor, Resize

class MyDataset(Dataset):
    def __init__(self, datapath, is_train, transform=None, is_base_train=True):
        self.class_name = ['capacitor', 'fuente_corriente', 'fuente_voltaje_AC', 'fuente_voltaje_DC', 'inductor', 'resistencia', 'tierra']
        self.is_base_train = is_base_train
        self.image_list_path = []
        self.label_list_path = []
        self.categories = []

        if is_train:
            path_data = [os.path.join(datapath, "train")]
        else:
            path_data = [os.path.join(datapath, "test")]

        image_forder_path = os.path.join(path_data[0], os.listdir(path_data[0])[0])
        label_forder_path = os.path.join(path_data[0], os.listdir(path_data[0])[1])
        for image_name in os.listdir(image_forder_path):
            label_name = image_name.replace(".jpg", ".txt")
            self.image_list_path.append(os.path.join(image_forder_path, image_name))
            self.label_list_path.append(os.path.join(label_forder_path, label_name))

        self.transform = transform


    def __len__(self):
        return len(self.image_list_path)

    def __getitem__(self, idx):
        image_path = self.image_list_path[idx]
        label_path = self.label_list_path[idx]

        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        if self.transform:
            image = self.transform(image)

        bboxs = []
        labels = []
        with open(label_path, "r") as label_file:
            for line in label_file:
                if line.strip() == "":
                    continue
                line = line.strip().split()
                box = [float(line[1])*w, float(line[2])*h, float(line[3])*w, float(line[4])*h]
                xmin = box[0] - box[2]/2
                ymin = box[1] - box[3]/2
                xmax = box[0] + box[2]/2
                ymax = box[1] + box[3]/2
                if self.is_base_train:
                    if int(line[0]) not in [1, 2, 6]:
                        bboxs.append([int(xmin), int(ymin), int(xmax), int(ymax)])
                        labels.append(int(line[0]))
                else:
                    bboxs.append([int(xmin), int(ymin), int(xmax), int(ymax)])
                    labels.append(int(line[0]))
        target = {
            "boxes": torch.tensor(bboxs, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }
            
        return image, target
        

if __name__ == '__main__':
    path = "dataset/ElementosCircuitoElectrico_2"
    transform = Compose([ToTensor()])

    dataset = MyDataset(path, True, transform)
    image, target = dataset.__getitem__(10)
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

    model.train()
    loss = model([image], [target])
    print(loss)