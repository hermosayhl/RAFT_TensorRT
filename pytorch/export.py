import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder



DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey()


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            # flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            # viz(image1, flow_up)

            onnx_path = "./RAFT.onnx"
            torch.onnx.export(
                model,
                (image1, image2),
                onnx_path,
                input_names=["image1", "image2"],
                output_names=["flow"],
                opset_version=11,
                do_constant_folding=True,
                dynamic_axes={
                    "image1": {0: "image1_batch", 2: "image1_height", 3: "image1_width"},
                    "image2": {0: "image2_batch", 2: "image2_height", 3: "image2_width"},
                    "flow"  : {0: "flow_batch",   2: "flow_height",   3: "flow_width"}
                }
            )
            print("export model to {}".format(onnx_path))
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="./pretrained/raft-sintel.pth", help="restore checkpoint")
    parser.add_argument('--path', default="./demo-frames", help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
