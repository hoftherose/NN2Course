from pathlib import Path
import torch
import torchvision
import torchvision.utils as utils
import PIL
from PIL import Image

def Unsqueezer(img):
  return img.unsqueeze(0)

model = torch.load("torchM")
model.eval()

def Forward(inputImgName: str):
    outputImgName = ".".join(inputImgName.split(".")[:-1]) + "-seg.pbm"
    path = Path("Figaro-1k/Original/Testing/") #change this
    ImageTransforms = torchvision.transforms.Compose([torchvision.transforms.Resize((500,500)),
                                                      torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
                                                      Unsqueezer])

    img = PIL.Image.open(path/inputImgName)
    img_size = img.size
    img = ImageTransforms(img).cuda()

    res = torch.sigmoid(model(img))
    res = res.argmax(dim=1)
  
    to_pil = torchvision.transforms.ToPILImage()
    p_img = to_pil(res.detach().cpu().type(torch.ByteTensor))
    p_img = p_img.resize(img_size)
    p_img.size
  
    p_img.save(outputImgName)
    return outputImgName

inname = "Frame00018-org.jpg"
outname = Forward(inname)
