import os.path, glob
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random

class UnalignedDataset(BaseDataset):
    def __init__(self, opt):
        super(UnalignedDataset, self).__init__()
        self.opt = opt
        self.transform = get_transform(opt)

        datapath = os.path.join(opt.dataroot, opt.phase + '*')
        self.dirs = sorted(glob.glob(datapath))

        self.paths = [sorted(make_dataset(d)) for d in self.dirs]
        self.sizes = [len(p) for p in self.paths]

    def load_image(self, dom, idx):
        path = self.paths[dom][idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, path

    def __getitem__(self, index):
        if not self.opt.isTrain:
            if self.opt.serial_test:
                for d,s in enumerate(self.sizes):
                    if index < s:
                        DA = d; break
                    index -= s
                index_A = index
            else:
                DA = index % len(self.dirs)
                index_A = random.randint(0, self.sizes[DA] - 1)
        else:
            # Choose two of our domains to perform a pass on
            # DA - clean, DB - badweather
            DA = 0
            index_A = random.randint(0, self.sizes[DA] - 1)

        A_img, A_path = self.load_image(DA, index_A)
        bundle1 = {'A': A_img, 'DA': DA, 'path': A_path}
        bundle2 = {'A': A_img, 'DA': DA, 'path': A_path}
        bundle3 = {'A': A_img, 'DA': DA, 'path': A_path}

        bundle = [bundle1, bundle2, bundle3]

        if self.opt.isTrain:
            for i in range(self.opt.badweather_domains): # 0,1,2 代表bad weather的三个域
                index_B = random.randint(0, self.sizes[i + 1] - 1)
                B_img, _ = self.load_image(i + 1, index_B)
                bundle[i].update( {'B': B_img, 'DB': i + 1} )

        return bundle

    def __len__(self):
        if self.opt.isTrain:
            return max(self.sizes)
        return sum(self.sizes)

    def name(self):
        return 'UnalignedDataset'