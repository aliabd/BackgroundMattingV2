import torch
from torchvision.transforms.functional import to_tensor, to_pil_image
import gdown
import gradio
import os

if not os.path.exists("model.pth"):
    gdown.download("https://drive.google.com/uc?id=1-t9SO--H4WmP7wUl1tVNNeDkq47hjbv4", "model.pth")

model = torch.jit.load('model.pth').eval()


def image_matting(src, bgr):
    src = to_tensor(src).unsqueeze(0)
    bgr = to_tensor(bgr).unsqueeze(0)
    if src.size(2) <= 2048 and src.size(3) <= 2048:
        model.backbone_scale = 1 / 4
        model.refine_sample_pixels = 80_000
    else:
        model.backbone_scale = 1 / 8
        model.refine_sample_pixels = 320_000
    pha, fgr = model(src, bgr)[:2]
    com = pha * fgr + (1 - pha) * torch.tensor([120 / 255, 255 / 255, 155 / 255], device='cpu').view(1, 3, 1, 1)
    return to_pil_image(com[0].cpu()), to_pil_image(pha[0].cpu()), to_pil_image(fgr[0].cpu())

title = "Real-Time High-Resolution Background Matting"
description = "This is a demo of Background Matting V2. To use it upload your own images or click the example below. " \
              "This model requires both a source image and a background image. For more information, see the links at the bottom."
examples = [
    ["example_images/src.png","example_images/bgr.png"]
]

article = """
<p style='text-align: center'>This demo is based on the <a href="https://arxiv.org/abs/2012.07810">Real-Time High-Resolution Background Matting</a> paper</p>
<p style='text-align: center'>For more info see the: <a href="https://grail.cs.washington.edu/projects/background-matting-v2/">Project Site</a> | 
<a href="https://www.youtube.com/watch?v=oMfPTeYDF9g">Project Video</a> | <a href="https://github.com/PeterL1n/BackgroundMattingV2">Github Repo</a> </p>
<p style='text-align: center'>This work is licensed under the <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode">Creative Commons Attribution NonCommercial ShareAlike 4.0 License</a></p>
"""

inputs = [gradio.inputs.Image(type="pil", label="Source Image"),gradio.inputs.Image(type="pil", label="Background Image")]
outputs = [gradio.outputs.Image(label="Composite"), gradio.outputs.Image(label="Alpha"), gradio.outputs.Image(label="Foreground")]
gradio.Interface(image_matting, inputs, outputs, title=title, description=description, examples=examples,
                 article=article, allow_flagging=False).launch()