import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from ._ext import crop_and_resize as _backend


class CropAndResizeFunction(Function):

    def __init__(self, crop_height, crop_width, extrapolation_value=0):
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.extrapolation_value = extrapolation_value

    def forward(self, image, boxes, box_ind):
        crops = torch.zeros_like(image)

        if image.is_cuda:
            _backend.crop_and_resize_gpu_forward(
                image, boxes, box_ind,
                self.extrapolation_value, self.crop_height, self.crop_width, crops)
        else:
            _backend.crop_and_resize_forward(
                image, boxes, box_ind,
                self.extrapolation_value, self.crop_height, self.crop_width, crops)

        # save for backward
        self.im_size = image.size()
        self.save_for_backward(boxes, box_ind)
        print("Finish Crop And Resize ")
        print(crops)
        return crops

    def backward(self, grad_outputs):
        boxes, box_ind = self.saved_tensors

        grad_outputs = grad_outputs.contiguous()
        grad_image = torch.zeros_like(grad_outputs).resize_(*self.im_size)

        if grad_outputs.is_cuda:
            _backend.crop_and_resize_gpu_backward(
                grad_outputs, boxes, box_ind, grad_image
            )
        else:
            _backend.crop_and_resize_backward(
                grad_outputs, boxes, box_ind, grad_image
            )

        return grad_image, None, None


class CropAndResize(nn.Module):
    """
    Crop and resize ported from tensorflow
    See more details on https://www.tensorflow.org/api_docs/python/tf/image/crop_and_resize
    """

    def __init__(self, crop_height, crop_width, extrapolation_value=0):
        super(CropAndResize, self).__init__()

        self.crop_height = crop_height
        self.crop_width = crop_width
        self.extrapolation_value = extrapolation_value

    def forward(self, image, boxes, box_ind):
        return CropAndResizeFunction(self.crop_height, self.crop_width, self.extrapolation_value)(image, boxes, box_ind)


if __name__ == "__main__":
    #Test Module
    import numpy as np
    from torch.autograd import Variable

    # xxxx = xxx.repeat(4,1)
    # print(xxxx)
    rs = np.random.random(4)*0.05
    print(rs)
    bs = 256
    hw = 8
    sb = 8
    x = Variable(torch.Tensor(np.arange(bs*64*hw*hw, dtype=np.float32).reshape((bs,64,hw,hw)))) / 20.0
    rs = np.random.random(4) * 0.05
    bs = x.size(0)
    bbox = torch.Tensor([rs[0], rs[1], 1 - rs[2], 1 - rs[3]])

    # print(bottom.size(0))
    # print(bottom)
    # x = CropAndResizeFunction(sb,sb)(bottom, torch.Tensor([[0.4561, 0.3417, 0.7110, 0.8990], [0,0,0.9,0.9]]), torch.IntTensor(list(range(bs))))
    # y = CropAndResizeFunction(sb, sb)(bottom, torch.Tensor([[0.4561, 0.3417, 0.7110, 0.8990], [0, 0, 0.9, 0.9], [0.2,0.2,1,1]]),
    #                                   torch.Tensor([0, 0]))
    bbox = bbox.repeat(bs, 1)
    x = CropAndResizeFunction(sb, sb)(x, bbox, torch.IntTensor(list(range(bs))))
    # z = torch.cat((x,y), 0)
    print(x.size())
    # print(z.size())
    # print(x)
    pass