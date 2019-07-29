# -*- coding: utf-8 -*-
"""
@author: Prabhu <prabhu.appalapuri@gmail.com>
"""
import os
import torch
import torch.nn as nn
import torchvision.transforms as transform
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.datasets as dset
import torchvision.utils as vutils
import numpy as np
from src.generator import Gen
from src.discriminator import Disc
import matplotlib.animation as annime
from IPython.display import HTML
import pickle
import argparse

def get_args():
    parser = argparse.ArgumentParser("""Train DCGAN for creating fake images""")
    parser.add_argument('-e', "--num_epochs", type=int, default= 35)
    args = parser.parse_args()
    return args

torch.manual_seed(546)

image_size = 64  # using image size as 64x64 with 3 channel i.e 3x64x64
total_epochs = 15
# path = 'results/performance.txt'
dataset = dset.ImageFolder(root = 'data/celeba', transform = transform.Compose([transform.Resize(image_size),
                                                                            transform.CenterCrop(image_size),
                                                                            transform.ToTensor(),
                                                                            transform.Normalize((0.5,0.5,0.5),
                                                                                                (0.5,0.5,0.5))]))

datasetLoader = DataLoader(dataset=dataset, batch_size=128, shuffle=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
sample  = iter(datasetLoader).__next__()
# print(vutils.make_grid(sample[0].to(device)[:64], padding= 2, normalize= True).cpu().size())
# print(sample)
plt.figure(figsize=(8,8))
plt.axis('off')
plt.title("Train set images")
plt.imshow(np.transpose(vutils.make_grid(sample[0].to(device)[:64], padding= 2, normalize= True).cpu(),(1,2,0)))
plt.savefig('fig/sample_image.png')
plt.close()

# initialize weights for Generator and Discriminator networks
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') !=-1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def plot_fig(Gen_losses, Dis_losses):
    plt.figure(figsize=(10,8))
    plt.title("Generator and Discrimator loss while training")
    plt.plot(Gen_losses, label='G_loss')
    plt.plot(Dis_losses, label = 'D_loss')
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('fig/Loss.png')
    return plt.draw()


def train(save_performance, opt):

    model_G = Gen(num_GenFeatureMaps = 64, num_OutputChannels = 3, vector_size = 100).to(device)
    model_D = Disc(num_Channels = 3, num_DisFeaturesMaps = 64, vector_size = 100).to(device)

    model_G.apply(weights_init)
    model_D.apply(weights_init)

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(64, 100, 1,1, device=device)
    real_imgLable = 1
    fake_imgLabel = 0

    optimizerD = optim.Adam(model_D.parameters(), lr= 0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(model_G.parameters(), lr= 0.0002, betas=(0.5, 0.999))

    img_list = []
    G_loss = []
    D_loss = []
    iters = 0

    for epoch in range(opt.num_epochs):
        for i, data in enumerate(datasetLoader):
            """
            Stage 1: The Discriminator network will be trained on real data and also on fake data, which is generated by 
            Generator network. After evaluation of all gradients w.r.t real and fake data, the Discriminator model will be 
            optimized.
            """

            model_D.zero_grad()
            # Stage 1.1 : training from a real input
            real_data = data[0].to(device)
            realSize = real_data.shape[0]
            labels = torch.full((realSize,), real_imgLable, device= device) # torch.Size([128])
            # print(labels)
            real_predictions= model_D(real_data)   # torch.Size([128, 1, 1, 1])
            real_predictions.view(-1)  # torch.Size([128])
            real_errD = criterion(real_predictions, labels)
            real_errD.backward()
            D_x = real_predictions.mean().item()

            # Stage 1.2 : training from a fake input
            # 100 means latent vector size, which is the same input value for Generator model
            noise = torch.randn(realSize, 100, 1,1, device= device)
            fake_data= model_G(noise)
            labels.fill_(fake_imgLabel)
            # print(labels)
            fake_predictions=model_D(fake_data.detach())  #torch.Size([128, 1, 1, 1])
            fake_predictions.view(-1)  #torch.Size([128])
            fake_errD = criterion(fake_predictions, labels)
            fake_errD.backward()
            D_G_z1 = fake_predictions.mean().item()
            errD = real_errD + fake_errD
            optimizerD.step()

            """
            stage 2 : which is for updating Generator network after Discriminator network error evaluation D(G(z)) 
            i.e maximization of log(D(G(z))) in order to increase more realistic images from Generator. The more 
            Discriminator network confuses to identify between real and fake then the more error will yield.  
            """
            model_G.zero_grad()
            labels.fill_(real_imgLable)
            # print('********')
            # print(fake_data.size())
            predictionsG = model_D(fake_data)
            predictionsG.view(-1)
            # print(predictionsG.view(-1))
            errG = criterion(predictionsG, labels)
            errG.backward()
            D_G_z2 = errG.mean().item()
            optimizerG.step()

            print("Iter:[{}/{}]\tEpoch:[{}/{}]\tLossD:{}\tLossG:{}\tD(X):{}\tD(G(z)):{}/{}".format(i+1,len(datasetLoader), epoch+1,opt.num_epochs, errD.item(), errG.item(),D_x, D_G_z1, D_G_z2))
            if i+1 % len(datasetLoader) == 0:
                with open(save_performance, 'a') as f:
                    f.write("Iter:[{}/{}]\tEpoch:[{}/{}]\tLossD:{}\tLossG:{}\tD(X):{}\tD(G(z)):{}/{}\n".format(i+1,len(datasetLoader), epoch+1,opt.num_epochs, errD.item(), errG.item(),D_x, D_G_z1, D_G_z2))

            G_loss.append(errG.item())
            D_loss.append(errD.item())

            # print fake images with a frequency of 500 iterations of update to Generator network weights
            if (iters% 500 ==0) or (epoch+1 == opt.num_epochs) or (i+1==len(datasetLoader)):
                with torch.no_grad():
                    fake_image = model_G(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake_image, padding=2,normalize=True))
                iters += 1
        torch.save(model_G, 'models/generator_model')
        torch.save(model_D, 'models/discriminator_model')
        plot_fig(Gen_losses=G_loss, Dis_losses=D_loss)
    losses ={}
    losses['G_loss']=G_loss
    losses['D_loss']=D_loss
    with open('results/losses.pkl','wb') as f:
        pickle.dump(losses, f)
    return img_list

if __name__ == '__main__':
    opt= get_args()
    img_list = train(save_performance='results/performance.txt', opt=opt)
    with open('results/fake_images.pkl', 'wb') as f:
        pickle.dump(img_list, file=f)
    # visualize the fake images
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    img = [[plt.imshow(np.transpose(i, (1,2,0)),animated= True)] for i in img_list]
    animation_ = annime.ArtistAnimation(fig, img, interval= 1000, repeat_delay= 1000, blit=True)
    animation_.save('fig/animation.gif', writer= 'imagemagick',fps=60)
    plt.show()
    print(HTML(animation_.to_jshtml()))





