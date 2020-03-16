"""
@author Feiyang Cai
@email feiyang.cai@vanderbilt.edu
@create date 2020-03-16 13:35:30
@modify date 2020-03-16 13:35:30
@desc [description]
"""
import torch
from PIL import Image
from torchvision import transforms
from scripts.network import VAEPerceptionNet
import numpy as np

image_path = "../robust_vae/data/test/in/40.png"
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
image = Image.open(image_path).convert("RGB")
image = transform(image)
model = VAEPerceptionNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.load_state_dict(torch.load("./models/perception.pt", map_location=device))
model.eval()

image = image.view([1, image.shape[0], image.shape[1], image.shape[2]])
x, _, _, _, _, _ = model(image)
reconstructed_image = x[0].cpu().data.numpy()
reconstructed_image = np.rollaxis(reconstructed_image, 0, 3)
reconstructed_image = Image.fromarray(np.uint8(reconstructed_image*255.0))
reconstructed_image.show()
