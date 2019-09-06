import model
import torch
import torchvision
import PIL.Image as Image
import argparse
from torchvision.transforms import Resize,ToTensor,Normalize
import torch.nn.functional as F


def argparser():

    arg = argparse.ArgumentParser(description="test style transforms")
    arg.add_argument("--image",type=str,help="input image")
    arg.add_argument("--out_name",type=str,help="out style image name")
    arg.add_argument("--model",type=str,default='./checkpoint/checkpoint_245.pth')
    arg.add_argument("--rato",type=float,default=0.7,help="rate of ori image")
    arg.add_argument("--size",type=int,default=256,help="resize ")
    return arg.parse_args()

def main():
    arg = argparser()

    genrate_model = model.genrate_model()
    genrate_model.load_state_dict(torch.load(arg.model))

    cuda = torch.cuda.is_available()
    if cuda:
        genrate_model.cuda()

    genrate_model.eval()

    image = Image.open(arg.image)
    ori_h,ori_w = image.size
    image = Resize((arg.size,arg.size))(image)
    image = ToTensor()(image)
    image = Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(image)
    image = image.unsqueeze(0)
    if cuda:
        image = image.cuda()

    with torch.no_grad():
        genrate_image = genrate_model(image)
        out_image = (1 - arg.rato) * (image * 0.5 + 0.5) + arg.rato * (genrate_image * 0.5 + 0.5)
        out_image = F.interpolate(out_image,(ori_w,ori_h))
        torchvision.utils.save_image(out_image,arg.out_name)




if __name__=='__main__':
    main()