import os
import os.path as osp
import glob
import open3d as o3d
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_txt_array
import numpy as np




class UnevenGroundDataset(InMemoryDataset):


    category_ids = {
    'traversable': '0',
    'nontraversable': '1',
    }

    seg_classes = {
        'traversable': [0],
        'nontraversable': [1],
    }

    def __init__(
        self,
        root,
        categories=None,
        split="train",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        print("mm Intializing UnevenGroundDataset dataset")
        self.root = root

        if categories is None:
            categories = list(self.category_ids.keys())

        self.categories = categories

        self.y_mask = torch.zeros((len(self.seg_classes.keys()), 2),
                                  dtype=torch.bool)


        super(UnevenGroundDataset, self).__init__(
            root, transform, pre_transform, pre_filter
        )
        path = self.processed_paths[0]
        self.data, self.slices = torch.load(path)

        for i, labels in enumerate(self.seg_classes.values()):
            self.y_mask[i, labels] = 1
    
    @property
    def num_classes(self):
        return self.y_mask.size(-1)

    @property
    def raw_file_names(self):
        return ["uneven_0.pcd"]

    @property
    def processed_file_names(self):
        return ["processed_dataset.pt"]

    def download(self):
        print(
            "download function is void, makesure data is locally availabe and under provided root folder"
        )

    def process(self):

        filenames_list = glob.glob(self.root + "/*.pcd")
        print("all files are: ", filenames_list)

        train_data_list = []

        for filename in filenames_list:
            current_pcd_file = o3d.io.read_point_cloud(filename)
            points = current_pcd_file.points
            colors = current_pcd_file.colors

            labels = []
            for color in colors:
                # if color is red the label is "1" NON-TRAVERSABLE
                if color[0]:
                    labels.append(1)
                # if color i green label is "0" =TRAVERSABLE
                elif color[1]:
                    labels.append(0)
                # if none, just add -1 as label, to be ignored
                else:
                    labels.append(-1)
            
            labels = torch.tensor(labels).long()
            points = torch.tensor(points)
 
            data = Data(pos=points, y=labels)

            train_data_list.append(data)
            # test_data_list.append(data)
        print("saving data to : ", self.processed_paths[0])
        torch.save(self.collate(train_data_list), self.processed_paths[0])
