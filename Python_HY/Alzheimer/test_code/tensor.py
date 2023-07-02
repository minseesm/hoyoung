import torch
import numpy as np
from PIL import Image

print("==== Image Read ====")
img = Image.open("dataset/test/Mild Impairment/1 (2).jpg")  # python list
print("==== Image convert to Numpy array ====")
img = np.array(img)                                         # np.array
print(img)
print("==== Image convert to Tensor ====")
tensor_img = torch.tensor(img)                              # tensor
print(tensor_img)
print(tensor_img.shape)

# add/reduce dimention
tensor_img_add_dim = tensor_img.unsqueeze(0).unsqueeze(0)       # [1, 1, 128, 128]
print(tensor_img_add_dim.shape)
# tensor_img_add_dim = tensor_img_add_dim.squeeze()
# print(tensor_img_add_dim.shape)

# repeat
tensor_img_add_dim = tensor_img_add_dim.repeat(1, 3, 1, 1)     # [1, 3, 128, 128]
print(tensor_img_add_dim.shape)