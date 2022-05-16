from nyuv2 import NYUv2
from torchvision import transforms

t = transforms.Compose([transforms.ToTensor()])
NYUv2(root="datasets", download=True, 
      rgb_transform=t, seg_transform=t, depth_transform=t)