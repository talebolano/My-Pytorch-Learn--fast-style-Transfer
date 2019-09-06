import model
import dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import os
import config
import PIL.Image as Image
import torchvision.datasets
from torch.utils.data import DataLoader
from torchvision.transforms import Compose,ToTensor,Normalize,Resize,RandomHorizontalFlip
import apex
import logging

class Gram(nn.Module):
    def forward(self, input):
        b,c,h,w = input.size()
        F = input.view(b,c,h*w)
        G = torch.bmm(F,F.transpose(1,2))
        G.div_(h*w)
        return G

def main():


    if not os.path.exists(config.out_image):
        os.mkdir(config.out_image)
    if not os.path.exists(config.save_model):
        os.mkdir(config.save_model)
    writer = SummaryWriter()

    train_lists,val_lists = dataset.train_val_split(config.image_path,config.val,config.seed)
    #print(train_lists)

    train_transfoms = Compose([Resize((config.resize,config.resize)),
                               RandomHorizontalFlip(),
                               ToTensor(),
                               Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    val_transfoms = Compose([Resize((config.resize,config.resize)),
                             ToTensor(),
                               Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    train_datasets = dataset.dataloader(train_lists,train_transfoms)
    val_datasets = dataset.dataloader(val_lists,val_transfoms)

    train_dataloader = DataLoader(train_datasets,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  drop_last=True)

    val_dataloader = DataLoader(val_datasets,
                                batch_size=4,
                                shuffle=False,
                                num_workers=config.num_workers,
                                pin_memory=True)

    cuda = torch.cuda.is_available()
    genrate_model = model.genrate_model()
    D_model = model.style_model()

    if config.load_checkpoint:
        genrate_model.load_state_dict(torch.load(config.checkpoint))
    if cuda:
        genrate_model.cuda()
        D_model.cuda()


    #optim = torch.optim.sgd.SGD(genrate_model.parameters(),config.lr,)
    optim = torch.optim.Adam(genrate_model.parameters(),config.lr)

    style_criterion = torch.nn.MSELoss()
    constant_criterion = torch.nn.MSELoss()


    if cuda:
        style_criterion.cuda()
        constant_criterion.cuda()



########################style image###########################
    style_image = Image.open(config.style_image)
    style_image = Resize((config.resize,config.resize))(style_image)
    style_image = ToTensor()(style_image)
    style_image = Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(style_image)
    style_image =style_image.unsqueeze(0)
    if cuda:
        style_image = style_image.cuda()
    style_ori_image = [style_ori_image_.detach() for style_ori_image_ in D_model(style_image, 'style')]
    Gram_style = [Gram()(style_ori_image[a]) for a in range(4)]

    best_loss = float('inf')
########################################train##########################################
    for epoch in range(config.epochs):
        genrate_model.train()
        for i,image in enumerate(train_dataloader):
            optim.zero_grad()
            if cuda:
                image = image.cuda()
            genrate_image = genrate_model(image)

            constant_ori_image = D_model(image,'constant').detach()
            constant_genrate_image = D_model(genrate_image,'constant')


            constant_loss =constant_criterion(constant_ori_image,constant_genrate_image)

            #style_ori_image = [style_ori_image_.detach() for style_ori_image_ in D_model(style_image,'style')]
            style_genrate_image = [style_genrate_image_ for style_genrate_image_ in D_model(genrate_image,'style')]

            Gram_genrate = [Gram()(style_genrate_image[a]) for a in range(4)]

            style_losses = [config.style_weight*style_criterion(Gram_genrate[a],Gram_style[a].expand_as(Gram_genrate[a])) for a in range(4)]
            style_loss = sum(style_losses)

            loss = constant_loss+style_loss

            loss.backward()

            optim.step()
            print('epoch:{}  iter:{}  loss:{}'.format(epoch,i,loss))
            writer.add_scalar('scalar/loss',float(loss),epoch*len(train_dataloader)+i)
            if float(loss)<best_loss:
                best_loss =float(loss)
                torch.save(genrate_model.state_dict(), config.save_model + '/best.pth')


        if (epoch+1)%config.val_epoch==0:
            genrate_model.eval()
            with torch.no_grad():
                for i,image in enumerate(val_dataloader):
                    if cuda:
                        image = image.cuda()
                        genrate_image = genrate_model(image)
                        torchvision.utils.save_image(torch.cat([image*0.5+0.5,genrate_image*0.5+0.5]),config.out_image+'/{}.jpg'.format(i))
                        print('val_epoch:{}  iter:{}'.format(epoch, i))
            #torch.save(genrate_model.state_dict(),config.save_model+'/checkpoint_{}.pth'.format(epoch))


if __name__=="__main__":
    main()