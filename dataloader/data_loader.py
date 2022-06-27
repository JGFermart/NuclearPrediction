import random
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from .image_folder import make_dataset
from PIL import Image, ImageFile, ImageOps

ImageFile.LOAD_TRUNCATED_IMAGES = True


######################################################################################
# Create the dataloader
######################################################################################
class CreateDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.img_A, self.img_A_size = make_dataset(opt.A_file)
        self.img_M, self.img_M_size = make_dataset(opt.M_file)
        self.img_N, self.img_N_size = make_dataset(opt.N_file)
        params = get_params(opt)
        self.transform = get_transform(opt, params=params, convert=True, augment=False)

    def __len__(self):
        """return the total number of images in the dataset"""
        return self.img_A_size

    def name(self):
        return "cell"

    def __getitem__(self, item):
        """return a data point and its metadata information"""
        img_A, img_A_path = self._load_img(item, self.img_A)
        img_M, img_M_path = self._load_img(item, self.img_M)
        img_N, img_N_path = self._load_img(item, self.img_N)

        return {'img_A':img_A, 'img_M':img_M, 'img_N':img_N, 'img_path':img_A_path}

    def _load_img(self, item, path):
        img_path = path[item % self.img_A_size]
        img_pil = Image.open(img_path).convert('RGB')
        img = self.transform(img_pil)
        img_pil.close()
        return img, img_path


def dataloader(opt):
    datasets = CreateDataset(opt)
    dataset = data.DataLoader(datasets, batch_size=opt.batch_size, shuffle=not opt.no_shuffle, num_workers=int(opt.nThreads), drop_last=True)

    return dataset


######################################################################################
# Basic image preprocess function
######################################################################################
def get_params(opt):
    w, h = opt.load_size, opt.load_size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.fine_size))
    y = random.randint(0, np.maximum(0, new_h - opt.fine_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


def _make_power_2(img, power, method=Image.BICUBIC):
    """resize the image to the size of log2(base) times"""
    ow, oh = img.size
    base = 2 ** power
    nw, nh = int(max(1, round(ow / base)) * base), int(max(1, round(oh / base)) * base)
    if nw == ow and nh == oh:
        return img
    return img.resize((nw, nh), method)


def _random_zoom(img, target_width, method=Image.BICUBIC):
    """random resize the image scale"""
    zoom_level = np.random.uniform(0.8, 1.0, size=[2])
    ow, oh = img.size
    nw, nh = int(round(max(target_width, ow * zoom_level[0]))), int(round(max(target_width, oh * zoom_level[1])))
    return img.resize((nw, nh), method)


def _scale_shortside(img, target_width, method=Image.BICUBIC):
    """resize the short side to the target width"""
    ow, oh = img.size
    shortsize = min(ow, oh)
    scale = target_width / shortsize
    return img.resize((round(ow * scale), round(oh * scale)), method)


def _scale_longside(img, target_width, method=Image.BICUBIC):
    """resize the long side to the target width"""
    ow, oh = img.size
    longsize = max(ow, oh)
    scale = target_width / longsize
    return img.resize((round(ow * scale), round(oh * scale)), method)


def _scale_randomside(img, target_width, method=Image.BICUBIC):
    """resize the side to the target width with random side"""
    if random.random() > 0.5:
        return _scale_shortside(img, target_width, method)
    else:
        return _scale_longside(img, target_width, method)


def _crop(img, pos=None, size=None):
    """crop the image based on the given pos and size"""
    ow, oh = img.size
    if size is None:
        return img
    nw = min(ow, size)
    nh = min(oh, size)
    if (ow > nw or oh > nh):
        if pos is None:
            x1 = np.random.randint(0, int(ow-nw)+1)
            y1 = np.random.randint(0, int(oh-nh)+1)
        else:
            x1, y1 = pos
        return img.crop((x1, y1, x1 + nw, y1 + nh))
    return img


def _pad(img):
    """expand the image to the square size"""
    ow, oh = img.size
    size = max(ow, oh)
    return ImageOps.pad(img, (size, size), centering=(0, 0))


def _flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def get_transform(opt, params=None, method=Image.BICUBIC, convert=True, augment=False):
    """get the transform functions"""
    transforms_list = []
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transforms_list.append(transforms.Resize(osize))
    elif 'scale_shortside' in opt.preprocess:
        transforms_list.append(transforms.Lambda(lambda img: _scale_shortside(img, opt.load_size, method)))
    elif 'scale_longside' in opt.preprocess:
        transforms_list.append(transforms.Lambda(lambda img: _scale_longside(img, opt.load_size, method)))
    elif "scale_randomside" in opt.preprocess:
        transforms_list.append(transforms.Lambda(lambda img: _scale_randomside(img, opt.load_size, method)))

    if 'zoom' in opt.preprocess:
        transforms_list.append(transforms.Lambda(lambda img: _random_zoom(img, opt.load_size, method)))

    if 'crop' in opt.preprocess and opt.isTrain:
        if params is None or 'crop_pos' not in params:
            transforms_list.append(transforms.Lambda(lambda img: _crop(img, size=opt.fine_size)))
        else:
            transforms_list.append(transforms.Lambda(lambda img: _crop(img, pos=params['crop_pos'], size=opt.fine_size)))
    if 'pad' in opt.preprocess:
        transforms_list.append(transforms.Lambda(lambda img: _pad(img)))     # padding image to square

    transforms_list.append(transforms.Lambda(lambda img: _make_power_2(img, opt.data_powers, method)))

    if not opt.no_flip and opt.isTrain:
        if params is None or 'flip' not in params:
            transforms_list.append(transforms.RandomHorizontalFlip())
        elif 'flip' in params:
            transforms_list.append(transforms.Lambda(lambda img: _flip(img, params['flip'])))

    if augment and opt.isTrain:
        transforms_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2))

    if convert:
        transforms_list.append(transforms.ToTensor())

    return transforms.Compose(transforms_list)