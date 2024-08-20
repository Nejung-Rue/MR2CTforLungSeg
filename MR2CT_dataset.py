import json 
import SimpleITK as sitk
import numpy as np

import cv2
import elasticdeform
from torch.utils.data import Dataset


root_path = 'your/path'
dataset = 'datasetname'

class StudyDataset(Dataset):
    def __init__(self, type='training', json_path='MR2CT4Seg.json', subtype=None, deform=10, ch=3):
        self.data = []
        self.deform = deform # M
        self.ch = ch # 1 means copy 1 channels, 3 means use 3-channel diversity
        with open(root_path + f'/{dataset}/{json_path}', 'rt') as f:
            self.data = json.load(f)[type]

        if subtype:
            with open(root_path + f'/{dataset}/{json_path}', 'rt') as f:
                self.data.extend(json.load(f)[subtype])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # sorce: 0~1, target: -1~1
        source_norm_slice_filename = item['source_norm_slice']
        source_norm_img_filename = item['source_norm_img']
        source_edge_filename = item['source_edge']
        target_filename = item['target_norm']
        target_norm_le1_filename = item['target_norm_le1']
        target_norm_le2_filename = item['target_norm_le2']
        prompt = item['prompt']
        
        source_norm_slice = sitk.ReadImage(root_path + f'/{dataset}/' + source_norm_slice_filename)
        source_norm_img = sitk.ReadImage(root_path + f'/{dataset}/' + source_norm_img_filename)
        source_edge = cv2.imread(root_path + f'/{dataset}/' + source_edge_filename, 0)
        target = sitk.ReadImage(root_path + f'/{dataset}/' + target_filename)
        target_norm_le1 = sitk.ReadImage(root_path + f'/{dataset}/' + target_norm_le1_filename)
        target_norm_le2 = sitk.ReadImage(root_path + f'/{dataset}/' + target_norm_le2_filename)

        source_norm_slice = sitk.GetArrayFromImage(source_norm_slice).astype(np.float32)
        source_norm_img = sitk.GetArrayFromImage(source_norm_img).astype(np.float32)
        source_edge = source_edge.astype(np.float32)
        target = sitk.GetArrayFromImage(target).astype(np.float32)
        target_norm_le1 = sitk.GetArrayFromImage(target_norm_le1).astype(np.float32)
        target_norm_le2 = sitk.GetArrayFromImage(target_norm_le2).astype(np.float32)
        
        if type == 'training' and self.deform != 0:
            # Deformation only during training step
            
            # Elastic deformation
            displacement = np.random.randn(2,8,8)*self.deform
            source_norm_slice_deformed = elasticdeform.deform_grid(np.array(source_norm_slice), displacement, order=3, cval=0)
            source_norm_img_deformed = elasticdeform.deform_grid(np.array(source_norm_img), displacement, order=3, cval=0)
            source_edge_deformed = elasticdeform.deform_grid(np.array(source_edge), displacement, order=3, mode='nearest')
            target_deformed = elasticdeform.deform_grid(np.array(target), displacement, order=3, cval=-1)
            target_norm_le1_deformed = elasticdeform.deform_grid(np.array(target_norm_le1), displacement, order=3, cval=-1)
            target_norm_le2_deformed = elasticdeform.deform_grid(np.array(target_norm_le2), displacement, order=3, cval=-1)

            # clip range
            source_norm_slice = np.clip(source_norm_slice_deformed, 0, 1)
            source_norm_img = np.clip(source_norm_img_deformed, 0, 1)
            source_edge = np.clip(source_edge_deformed, 0, 255)
            target = np.clip(target_deformed, -1, 1)
            target_norm_le1 = np.clip(target_norm_le1_deformed, -1, 1)
            target_norm_le2 = np.clip(target_norm_le2_deformed, -1, 1)
        
        # Normalize edge.
        source_edge = source_edge / 255.0
        
        # Expand dims.add()
        source_norm_slice = np.expand_dims(source_norm_slice, axis=-1)
        source_norm_img = np.expand_dims(source_norm_img, axis=-1)
        target = np.expand_dims(target, axis=-1)
        target_norm_le1 = np.expand_dims(target_norm_le1, axis=-1)
        target_norm_le2 = np.expand_dims(target_norm_le2, axis=-1)
        source_edge = np.expand_dims(source_edge, axis=-1)
        
        # Make 3 channel
        if self.ch == 3:
            source = np.concatenate((source_norm_img,source_norm_slice,source_edge),axis=-1)
            target = np.concatenate((target,target_norm_le1,target_norm_le2),axis=-1)
        elif self.ch == 1:
            source = np.concatenate((source_norm_img,source_norm_img,source_norm_img),axis=-1)
            target = np.concatenate((target,target,target),axis=-1)

        return dict(jpg=target, txt=prompt, hint=source)
    
    
class TestDataset(Dataset):
    def __init__(self, type='test', json_path='MR2CT4Seg.json', ch=3):
        self.data = []
        self.ch = ch
        with open(root_path + f'/{dataset}/{json_path}', 'rt') as f:
            self.data = json.load(f)[type]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_norm_slice_filename = item['source_norm_slice'] 
        source_norm_img_filename = item['source_norm_img'] 
        source_edge_filename = item['source_edge'] 
        prompt = item['prompt']

        source_norm_slice = sitk.ReadImage(root_path + f'/{dataset}/' + source_norm_slice_filename)
        source_norm_img = sitk.ReadImage(root_path + f'/{dataset}/' + source_norm_img_filename)
        source_edge = cv2.imread(root_path + f'/{dataset}/' + source_edge_filename, 0)

        source_norm_slice = sitk.GetArrayFromImage(source_norm_slice).astype(np.float32)
        source_norm_img = sitk.GetArrayFromImage(source_norm_img).astype(np.float32)
        source_edge = source_edge.astype(np.float32)

        # Normalize edge.
        source_edge = source_edge / 255.0
        
        # Expand dims.add()
        source_norm_slice = np.expand_dims(source_norm_slice, axis=-1)
        source_norm_img = np.expand_dims(source_norm_img, axis=-1)
        source_edge = np.expand_dims(source_edge, axis=-1)
        
        # Make 3 channel
        if self.ch == 3:
            source = np.concatenate((source_norm_img,source_norm_slice,source_edge),axis=-1)
        elif self.ch == 1:
            source = np.concatenate((source_norm_img,source_norm_img,source_norm_img),axis=-1)

        return dict(hint=source, txt=prompt, name=source_norm_slice_filename.split('/')[-1])