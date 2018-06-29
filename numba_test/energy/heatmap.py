import cv2
import numpy as np
from PIL import Image
from torch.nn import functional as F
from torchvision import models, transforms

__all__ = ['heatmap']

_models_to_use = {}
_features_blobs = []


def _hook_feature(module, input, output):
    _features_blobs.append(output.data.numpy())


def _load_model(name='squeezenet'):
    if name in _models_to_use:
        return _models_to_use[name]
    if name == 'densenet':
        net = _models_to_use[name] = models.densenet169(pretrained=True)
    elif name == 'squeezenet':
        net = _models_to_use[name] = models.squeezenet1_1(pretrained=True)
    else:
        print('Unknown model name, using squeezenet')
        net = _models_to_use[name] = models.squeezenet1_1(pretrained=True)
    net._modules.get('features').register_forward_hook(_hook_feature)
    return net


def returnCAM(feature_conv, weight_softmax, class_idx, height, width):
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


_normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
_preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   _normalize
])


def heatmap(npimg,modelname='squeezenet'):
    net = _load_model(modelname)

    net.eval()

    #print(np.shape(features_blobs))
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())
    img_pil=Image.fromarray(np.uint8(npimg))
    img_tensor = _preprocess(img_pil)
    img_variable = img_tensor.unsqueeze(0)
    logit = net(img_variable)
    #print(np.shape(features_blobs))

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()
    height, width, _ = npimg.shape
    heatmap = returnCAM(_features_blobs[-1], weight_softmax, [idx[0]], height, width)
    return heatmap