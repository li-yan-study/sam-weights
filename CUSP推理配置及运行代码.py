"""为了获得年龄信息，我们使用在 IMDB-WIKI 数据集上预训练的年龄分类器。我们使用了 Rothe 等人从没有面部标志的单张图像中发布的对真实和表观年龄的深度期望中发布的模型。

要准备模型，您需要下载原始 caffe 模型并将其转换为 PyTorch 格式。我们使用 Vadim Kantorov 发布的转换器 caffemodel2pytorch。然后将 PyTorch 模型命名为 as 并将其放入文件夹 .dex_imdb_wiki.caffemodel.pt/models"""

## 1 首先需要下载预训练权重 分为LS和RR版本。  
## 2 然后下载HRFAE中DEX分类器权重（https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/dex_imdb_wiki.caffemodel），并使用caffemodel2pytorch（https://github.com/vadimkantorov/caffemodel2pytorch）转换成.pt格式
## 3 使用以下代码进行推理

import os
import time
import pickle

import torch
import torch.nn.functional as F

import PIL.Image
import numpy as np

# Custom modules
from training.networks import VGG, module_no_grad
import legacy
from torch_utils import misc
import dnnlib

# Paths to local files
weights_path = "path/to/pretrained.pkl"  # Replace with the actual path to pretrained.pkl
vgg_path = "path/to/dex_imdb_wiki.caffemodel.pt"  # Replace with the actual path to dex_imdb_wiki.caffemodel.pt
sample_images_path = "path/to/sample_images"  # Replace with the actual path to sample images folder
output_dir = "path/to/output_images"  # Replace with the desired output directory

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# Running configuration
FFHQ_LS_KEY = "lats"  # Model trained on LATS dataset
FFHQ_RR_KEY = "hrfae"  # Model trained on HRFAE dataset

# Choose one from above
KEY = FFHQ_LS_KEY  # [FFHQ_RR_KEY or FFHQ_LS_KEY]

# Config and image side
configs = {
    FFHQ_LS_KEY: dict(
        side=256,
        classes=(1, 8)),
    FFHQ_RR_KEY: dict(
        side=224,
        classes=(20, 65))
}

# CUDA device
device = torch.device('cuda', 0)  # Use 'cpu' if no GPU is available
img_side = configs[KEY]['side']
data_labels_range = configs[KEY]['classes']

# Read image filenames
filenames_batch = [
    os.path.join(sample_images_path, f)
    for f in next(iter(os.walk(sample_images_path)))[2]
    if f[-4:] == '.png'
]


def load_model(model_path, vgg_path, device):
    """
    Load the pre-trained model and DEX age classifier.
    """
    with open(model_path, 'rb') as f:
        contents = legacy.load_network_pkl(f)  # Deserialize weights and source code

    # Get exponential moving average model
    G_ema = contents['G_ema']

    # Load DEX VGG classifier
    vgg = VGG()
    vgg_state_dict = torch.load(vgg_path)
    vgg_state_dict = {k.replace('-', '_'): v for k, v in vgg_state_dict.items()}
    vgg.load_state_dict(vgg_state_dict)
    module_no_grad(vgg)  # Important: Disable gradients

    # Set classifier
    G_ema.skip_grad_blur.model.classifier = vgg
    G_ema = G_ema.to(device).eval().requires_grad_(False)

    return G_ema


def run_model(G, img, label, global_blur_val=None, mask_blur_val=None, return_msk=False):
    """
    Run the model to generate an aged image.
    """
    # Transform label to One Hot Encoding
    cls = torch.nn.functional.one_hot(
        torch.tensor(label),
        num_classes=G.attr_map.fc0.init_args[0]
    ).to(img.device)

    # Content encoder
    _, c_out_skip = G.content_enc(img)

    # Style encoder
    s_out = G.style_enc(img)[0].mean((2, 3))

    truncation_psi = 1
    truncation_cutoff = None
    s_out = G.style_map(s_out, None, truncation_psi, truncation_cutoff)

    # Age mapping
    a_out = G.attr_map(cls.to(s_out.device), None, truncation_psi, truncation_cutoff)

    # Interleave style and age embeddings
    w = G.__interleave_attr_style__(a_out, s_out)

    # Global blur
    for i, (f, _) in enumerate(zip(G.skip_transf, c_out_skip)):
        if f is not None:
            c_out_skip[i] = G._batch_blur(c_out_skip[i], blur_val=global_blur_val)

    # Masked blur
    cam = G.skip_grad_blur(img.float())
    msk = cam
    for i, (f, c) in enumerate(zip(G.skip_transf, c_out_skip)):
        if f is not None:
            im_size = c.size(-1)
            blur_c = G._batch_blur(c, blur_val=mask_blur_val)
            if msk.size(2) != im_size:
                msk = F.interpolate(msk, size=(im_size, im_size), mode='area')
            merged_c = c * msk + blur_c * (1 - msk)
            c_out_skip[i] = merged_c

    # Decoder
    img_out = G.image_dec(c_out_skip, w)

    if return_msk:
        to_return = (img_out, msk, cam) if G.learn_mask is not None else (img_out, None, None)
    else:
        to_return = img_out

    return to_return


# Image preprocessing
imgs = [np.array(PIL.Image.open(f).resize((img_side, img_side)), dtype=np.float32).transpose((2, 0, 1))
        for f in filenames_batch]
im_in_tensor = (torch.tensor(np.array(imgs)) / 256 * 2 - 1).to(device)  # Normalize to [-1, 1]

# Aging steps
steps = 4  # Number of aging steps
n_images = im_in_tensor.shape[0]
im_in_tensor_exp = im_in_tensor[:, None].expand([n_images, steps, *im_in_tensor.shape[1:]]).reshape([-1, *im_in_tensor.shape[1:]])
labels_exp = torch.tensor(np.repeat(np.linspace(*data_labels_range, steps, dtype=int)[:, None], n_images, 1).T.reshape(-1)).to(device)

# Load the model
G_ema = load_model(weights_path, vgg_path, device)

# Run inference
batch_size = 12
im_out_tensor_exp = torch.concat([
    run_model(
        G_ema,
        mini_im,
        mini_label,
        global_blur_val=0.2,  # CUSP global blur
        mask_blur_val=0.8     # CUSP masked blur
    ) for mini_im, mini_label in zip(
        im_in_tensor_exp.split(batch_size),
        labels_exp.split(batch_size)
    )
])
im_out_tensor = im_out_tensor_exp.reshape([-1, steps, *im_out_tensor_exp.shape[1:]])

# Save results to files
def to_uint8(im_tensor):
    """
    Convert tensor to uint8 image format.
    """
    im_tensor = (im_tensor.detach().cpu().numpy().transpose((1, 2, 0)) + 1) * (256 / 2)
    im_tensor = np.clip(im_tensor, 0, 255).astype(np.uint8)
    return im_tensor

for fname, im_in, im_out, age_labels in zip(
        filenames_batch, im_in_tensor, im_out_tensor,
        labels_exp.cpu().numpy().reshape(-1, steps)
):
    base_name = os.path.splitext(os.path.basename(fname))[0]  # Extract filename without extension
    for step, (age_label, im_step) in enumerate(zip(age_labels, im_out)):
        # Create subdirectory for the target age
        age_subdir = os.path.join(output_dir, f"a{age_label}")
        if not os.path.exists(age_subdir):
            os.makedirs(age_subdir, exist_ok=True)

        # Save the image with the same name as the input file
        output_filename = f"{base_name}.png"
        output_path = os.path.join(age_subdir, output_filename)
        PIL.Image.fromarray(to_uint8(im_step)).save(output_path)
        print(f"Saved: {output_path}")
