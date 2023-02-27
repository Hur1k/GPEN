from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import os
import shutil
from data import FaceScrub, CelebA
from model import Inversion
import torch.nn.functional as F
import torchvision.utils as vutils
import time
import numpy as np
from torch.autograd import grad


parser = argparse.ArgumentParser(
    description='Adversarial Model Inversion Demo')
parser.add_argument('--img-size', type=int, default=64, metavar='')
parser.add_argument('--batch-size', type=int, default=128, metavar='')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='')
parser.add_argument('--epochs', type=int, default=100, metavar='')
parser.add_argument('--lr', type=float, default=0.01, metavar='')
parser.add_argument('--momentum', type=float, default=0.5, metavar='')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1, metavar='')
parser.add_argument('--log-interval', type=int, default=10, metavar='')
parser.add_argument('--nc', type=int, default=1)
parser.add_argument('--ndf', type=int, default=128)
parser.add_argument('--ngf', type=int, default=128)
parser.add_argument('--nz', type=int, default=512)
parser.add_argument('--truncation', type=int, default=512)
parser.add_argument('--c', type=float, default=50.)
parser.add_argument('--num_workers', type=int, default=1, metavar='')


def main():
    args = parser.parse_args()
    print("================================")
    print(args)
    print("================================")

    os.makedirs('celeba_invimg', exist_ok=True)
    os.makedirs('facescrub_invimg', exist_ok=True)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}

    torch.manual_seed(args.seed)

    # 加载dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = CelebA('./dataset', transform=transform)
    test_set = FaceScrub('./dataset', transform=transform)
    # 加载data_loader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    INV = nn.DataParallel(Inversion(nc=args.nc, ngf=args.ngf, nz=args.nz,
                          truncation=args.truncation, c=args.c, data_size=args.img_size)).to(device)
    
    # 允许加载反演器
    isload = True
    if isload:
        path = './model_weights/inversion_64_mh.pth'
        try:
            checkpoint = torch.load(path, map_location='cpu')
            INV.load_state_dict(checkpoint['model'])
            epoch = checkpoint['epoch']
            best_cl_acc = checkpoint['best_recon_loss']
            print("=> loaded inversion checkpoint '{}' (epoch {}, loss {:.4f})".format(
                path, epoch, best_cl_acc))
        except:
            print("=> load inversion checkpoint '{}' failed".format(path))
            return


    with torch.no_grad():
        for features, labels in test_loader:
            reconstruction = INV(features)
            for i in range(features.shape[0]):
                vutils.save_image(reconstruction[i], f'facescrub_invimg/inversion_64_mh/{labels[i]}.png')
    
    # with torch.no_grad():
    #     for features, labels in train_loader:
    #         reconstruction = INV(features)
    #         for i in range(features.shape[0]):
    #             vutils.save_image(reconstruction[i], f'celeba_invimg/{labels[i]}.png')
    
    # with torch.no_grad():
    #     reconstruction = INV(torch.tensor([0.19607843,0.34509804,0.63921569,0.42152466,0.62352941,0.98823529,0.43921569,0.41552511,0.07843137,0.67058824,0.37254902,0.38164251,0.18431373,0.91372549,0.14509804,1.,0.10196078,0.3372549,0.81568627,0.49803922,0.48235294,0.98431373,0.05098039,0.01593625,0.16078431,0.11372549,0.71764706,0.47773279,0.01176471,0.83921569,0.34901961,0.56595745,0.08235294,0.61568627,0.35686275,1.,0.56470588,0.52941176,0.9372549,0.8516129,0.1372549,0.25098039,0.58039216,0.,0.7254902,0.53333333,0.82352941,1.,0.7372549,0.20784314,0.40784314,0.32984293,0.4745098,0.89411765,0.89803922,1.,0.36078431,0.09411765,0.18039216,0.73142857,0.81568627,0.12156863,0.11372549,1.,0.14901961,0.54901961,0.75294118,0.,0.5372549,0.18039216,0.90980392,0.,0.70588235,0.10588235,0.62352941,0.00549451,0.74117647,0.37254902,0.2745098,0.30687831,0.25098039,0.41176471,0.90588235,0.02189781,0.29019608,0.47843137,0.60784314,0.0070922,1.,0.99607843,0.85490196,0.1192053,0.73333333,0.98431373,0.28235294,0.,0.58039216,0.01176471,0.98431373,0.00436681,0.92156863,0.92941176,0.21176471,0.54237288,0.18039216,0.64705882,0.18823529,0.52226721,0.48235294,0.04313725,0.38823529,0.00395257,0.71764706,0.61176471,0.55686275,0.645,0.70196078,0.11372549,0.57254902,0.,0.36078431,0.03921569,0.00392157,0.00462963,0.48627451,0.01176471,0.2,0.00452489,0.24705882,0.04705882,0.64313725,1.,0.79215686,0.73333333,0.2627451,0.02283105,0.69411765,0.57647059,0.06666667,1.,0.53333333,0.37254902,0.69803922,0.36945813,0.09019608,0.58823529,0.48627451,0.50196078,0.21960784,0.67058824,0.91372549,0.49003984,0.67843137,0.40392157,0.60784314,0.53974895,0.37647059,0.81176471,0.14509804,1.,0.23921569,0.98823529,0.0745098,0.18867925,0.50588235,0.03529412,0.68627451,0.03870968,0.34117647,0.23137255,0.87058824,0.05960265,0.39215686,0.91764706,0.67843137,0.94964029,0.20392157,0.29019608,0.,0.67539267,0.70588235,0.84705882,0.6,1.,0.6,0.54509804,0.96078431,1.,0.11764706,0.64705882,0.06666667,0.77192982,0.68627451,0.95686275,0.92156863,0.00598802,0.75686275,0.32941176,0.66666667,0.00578035,0.38823529,0.36470588,0.73333333,0.28333333,0.30196078,0.06666667,0.25098039,0.68253968,0.59607843,0.38431373,0.40392157,0.95555556,0.63137255,0.57254902,0.91372549,0.0070922,0.76862745,0.78039216,0.85098039,0.96688742,0.04705882,0.14509804,0.89411765,0.82165605,0.53333333,0.27058824,0.26666667,0.,0.23529412,0.99215686,0.08235294,0.00421941,0.70196078,0.72941176,0.11764706,0.00403226,0.82745098,0.01960784,0.87058824,0.49011858,0.90196078,0.90980392,0.50980392,0.64824121,0.3372549,0.14509804,0.63529412,0.98536585,0.83137255,0.14901961,0.97647059,0.59447005,0.31764706,0.86666667,0.08235294,0.00452489,0.36470588,0.41176471,0.00784314,0.42600897,0.86666667,0.15294118,0.58039216,0.99543379,0.92941176,0.49019608,0.4627451,0.,0.05098039,0.32941176,0.18823529,1.,0.35686275,1.,0.69411765,0.49803922,0.98823529,0.08627451,0.22745098,0.49003984,0.14509804,0.38823529,0.91372549,0.46025105,0.99607843,0.94901961,0.32941176,0.56170213,0.54117647,0.93333333,0.52941176,1.,0.57254902,0.47058824,0.75294118,0.02580645,0.40392157,0.85490196,0.00784314,0.8951049,0.56078431,0.18431373,0.94117647,0.94964029,0.45490196,0.21568627,0.91372549,0.32984293,0.98823529,0.3372549,0.55294118,0.99465241,0.43137255,0.89411765,0.58431373,0.26857143,0.94117647,0.09019608,0.31764706,1.,0.20784314,0.94901961,0.2,0.00598802,0.61568627,0.20784314,0.80392157,0.74566474,0.68235294,0.21176471,0.52941176,0.70491803,0.70980392,0.74901961,0.27058824,0.67724868,0.29803922,0.89411765,0.15686275,0.95555556,0.50980392,0.70980392,0.98431373,0.91428571,0.05882353,0.70588235,0.6627451,0.125,0.75294118,0.90980392,0.78039216,0.81528662,0.14509804,0.45098039,0.59607843,0.55411255,0.12156863,0.56470588,0.77647059,0.54008439,0.07843137,0.3254902,0.80784314,0.98380567,0.67843137,0.85490196,0.47843137,0.50988142,0.15686275,0.09411765,0.5254902,0.33165829,0.75686275,0.10588235,0.43921569,0.98536585,0.12941176,0.71372549,0.16470588,0.6,0.62745098,0.49803922,0.21960784,0.00452489,0.94117647,0.36862745,0.61176471,1.,0.89411765,0.03529412,0.31372549,1.,0.52156863,0.55686275,0.32156863,0.38164251,0.34117647,0.23529412,0.48627451,0.36945813,0.9372549,0.14901961,0.0745098,0.49411765,0.16078431,0.93333333,0.23921569,1.,0.54117647,0.88627451,0.5254902,1.,0.9254902,0.39215686,0.1372549,1.,0.96078431,0.16078431,0.6,1.,0.19215686,0.67843137,0.64313725,1.,0.95686275,0.04313725,0.63921569,0.1048951,0.41176471,0.84313725,0.51764706,1.,0.99607843,0.05882353,0.19215686,0.67015707,0.96470588,0.39215686,0.54509804,1.,0.81960784,0.83137255,0.01568627,0.26285714,0.09411765,0.27843137,0.8627451,0.77192982,0.76470588,0.67058824,0.74509804,0.77245509,0.42745098,0.6627451,0.16078431,0.74566474,0.58039216,0.56470588,0.03137255,0.00549451,0.61960784,0.78431373,0.63921569,0.00529101,0.49019608,0.71372549,0.62745098,0.01481481,0.00784314,0.41176471,0.2745098,0.91489362,0.05882353,0.42745098,0.84313725,0.96688742,0.5372549,0.45882353,0.85098039,0.98089172,0.4627451,0.15686275,0.94509804,0.00431034,0.64313725,0.72941176,0.63529412,0.97046414,0.94117647,0.96862745,0.39607843,0.51612903,0.58039216,0.8,0.83529412,0.50396825,0.47843137,0.2,0.35294118,0.005,0.22745098,0.82745098,0.21568627,0.,0.49411765,0.10980392,0.97254902,0.00460829,0.07843137,0.94117647,0.71372549,0.40723982]))
    #     vutils.save_image(reconstruction, f'hahah.png')
            
main()