import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy import ndimage
from torchvision.ops import nms
from pprintpp import pprint


class MultiScale_TM():
    """
    asdsadsa
    """
    def __init__(self, drawing_path="image/drawings/4.png", rotations = [0, 90]):
        self.drawing=cv2.imread(drawing_path)
        self.rotations=rotations

    def scale_tm(self, symbol_path ="image/template/img.png" , is_symmetry = True, resize=[0.5, 1, 2]):
        if not is_symmetry:
            symmetry = [rotation +180 for rotation in self.rotations]
            self.rotations.extend(symmetry)
        
        img_gray = cv2.cvtColor(self.drawing, cv2.COLOR_BGR2GRAY)

        raw_symbol = cv2.imread(symbol_path, 0)
        raw_list_bbox = []
        list_score = []
        for rotatio in self.rotations:
            symbol = ndimage.rotate(raw_symbol, rotatio, reshape=True)
            for size in resize:
                w, h = int(symbol.shape[1]*size), int(symbol.shape[0]*size)
                resize_symbol = cv2.resize(symbol, (w, h), interpolation=cv2.INTER_NEAREST)
                self.imshow(resize_symbol)
                res = cv2.matchTemplate(img_gray, resize_symbol, cv2.TM_CCOEFF_NORMED)
                THRESHOLD = 0.5
                loc = np.where(res >= THRESHOLD)

                raw_list_bbox.extend([[float(x), float(y), float(x + w), float(y + h)] for (x, y) in zip(*loc[::-1])])
                list_score.extend([float(res[y, x]) for (x, y) in zip(*loc[::-1])])

        if len(raw_list_bbox) == 0:
            return self.drawing

        keep_indices = (
            nms(
                torch.tensor(raw_list_bbox, dtype=torch.float32),
                torch.tensor(list_score, dtype=torch.float32),
                0.3,
            )
            .numpy()
            .tolist()
        )
        list_bbox = [raw_list_bbox[i] for i in keep_indices]


        for bbox in list_bbox:
            cv2.rectangle(self.drawing, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 1)


        return self.drawing

    def imshow(self, img, figsize=(6, 6)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")
        ax.imshow(img)
        plt.show()




if __name__ == "__main__":
    scale = MultiScale_TM()
    image = scale.scale_tm("image/template/img_1.png", is_symmetry=False)
    scale.imshow(image)



