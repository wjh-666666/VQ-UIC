# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

mask_type={"HD":1,      # HD: Human divers
           "PF":2,      # PF: Plants/sea-grass
           "WR":3,      # WR: Wrecks/ruins
           "RO":4,      # RO: Robots/instruments
           "RI":5,      # RI: Reefs and invertebrates
           "FV":6,      # FV: Fish and vertebrates
           "SR":7       # SR: Sand/sea-floor (& rocks)
           }

def image_to_mask(rgb_mask):
    '''convert h*w*rgb into 1-7 array'''
    new_list=np.zeros([rgb_mask.shape[0],rgb_mask.shape[1]])
    for i in range(rgb_mask.shape[0]):
        for j in range(rgb_mask.shape[1]):
            binary_string = ''.join(str(int(x)) for x in rgb_mask[i][j])
            new_list[i][j] = int(binary_string, 2)
    return new_list

class ImageFolder(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split
        if split == "train":
            masksplitdir = Path(root) / "train_masks"
        else:
            masksplitdir = Path(root) / "test_masks"
        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = sorted([f for f in splitdir.iterdir() if f.is_file()])
        self.mask_samples = sorted([f for f in masksplitdir.iterdir() if f.is_file()])
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")
        mask = Image.open(self.mask_samples[index]).convert("RGB")
        mask = mask.resize(img.size, Image.ANTIALIAS)


        if self.transform:
            to_tensor_transform = transforms.ToTensor()
            img_tensor = to_tensor_transform(img)
            mask_tensor = to_tensor_transform(mask)

            # 连接两个张量，假设它们具有相同的高度和宽度
            # 如果它们有不同的通道数，你可能需要调整它们以匹配形状
            #print(self.samples[index], self.mask_samples[index])
            #print(img_tensor.shape,maskimg_tensor.shape)
            combined_tensor = torch.cat((img_tensor, mask_tensor), dim=0)

            combined_tensor = self.transform(combined_tensor)
            # 根据通道数分离张量
            img = combined_tensor[:3, :, :]
            mask = combined_tensor[3:, :, :]

            mask = np.array(mask)
            mask = np.transpose(mask, (1, 2, 0))
            mask = image_to_mask(mask)
            human_mask = np.where(mask == mask_type["HD"], 1, 0)
            plant_mask = np.where(mask == mask_type["PF"], 1, 0)
            wreck_mask = np.where(mask == mask_type['WR'], 1, 0)
            robot_mask = np.where(mask == mask_type['RO'], 1, 0)
            reef_mask = np.where(mask == mask_type['RI'], 1, 0)
            fish_mask = np.where(mask == mask_type['FV'], 1, 0)
            sand_mask = np.where(mask == mask_type['SR'], 1, 0)
            masks = np.array([human_mask,
                              plant_mask,
                              wreck_mask,
                              robot_mask,
                              reef_mask,
                              fish_mask,
                              sand_mask
                              ])

            mask = torch.from_numpy(masks).float()
            return img, mask
        return img, mask

    def __len__(self):
        return len(self.samples)
