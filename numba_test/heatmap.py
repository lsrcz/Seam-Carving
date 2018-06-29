from torchvision import models, transforms
from torch.nn import functional as F
import torch
from PIL import Image
import numpy as np
import cv2
# networks such as googlenet, resnet, densenet already use global average pooling at the end, so CAM could be used directly.
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.numpy())
def returnCAM(feature_conv, weight_softmax, class_idx,height, width):
    # generate the class activation maps upsample to 256x256
    size_upsample = (width, height)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])


def heatmap(npimg):
    net = models.densenet161(pretrained=True)
    finalconv_name = 'features'
    net.eval()
    net._modules.get(finalconv_name).register_forward_hook(hook_feature)
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())
    img_pil=Image.fromarray(np.uint8(npimg))
    img_tensor = preprocess(img_pil)
    img_variable = img_tensor.unsqueeze(0)
    logit = net(img_variable)

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()
    height, width, _ = npimg.shape
    heatmap = returnCAM(features_blobs[0], weight_softmax, [idx[0]],height,width)
    return heatmap