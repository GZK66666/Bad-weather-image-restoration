import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

# def get_transform(opt):
#     transform_list = []
#     if 'resize' in opt.resize_or_crop:
#         transform_list.append(transforms.Resize(opt.load_size, Image.BICUBIC))
#
#     if opt.isTrain:
#         if 'crop' in opt.resize_or_crop:
#             transform_list.append(transforms.RandomCrop(opt.crop_size))
#         if not opt.no_flip:
#             transform_list.append(transforms.RandomHorizontalFlip())
#
#     transform_list += [transforms.ToTensor(),
#                        transforms.Normalize((0.5, 0.5, 0.5),
#                                             (0.5, 0.5, 0.5))]
#     return transforms.Compose(transform_list)

def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt.resize_or_crop:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))

    if 'crop' in opt.resize_or_crop:
        transform_list.append(transforms.RandomCrop(opt.crop_size))

    if not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)