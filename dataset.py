import cv2
import torch
import glob
import PIL.Image as Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

def train_val_split(image_path,val=0.25,seed=42):
    paths = glob.glob(image_path+'/*')
    train,val = train_test_split(paths,test_size=val,random_state=seed)
    return train,val

class dataloader(Dataset):
    def __init__(self,image_paths,transform_fn=None):
        super(dataloader,self).__init__()
        self.transform_fn = transform_fn
        self.paths = image_paths


    def __getitem__(self, index):
        image_path = self.paths[index]

        image = Image.open(image_path)
        if self.transform_fn:
            image = self.transform_fn(image)
        return image

    def __len__(self):
        return len(self.paths)
