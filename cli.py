import argparse

import cv2
import numpy as np

_MODELS = [
    "canny",
]


def available_models():
    """Returns the names of available models"""
    return _MODELS


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model", default="canny", choices=available_models(), help="name of the model")
parser.add_argument("--export-controlnet", action='store_true')
parser.add_argument("--export-diffusion-model", action='store_true')
parser.add_argument("--export-autoencoder", action='store_true')
args = parser.parse_args().__dict__

model_name: str = args.pop("model")
export_controlnet: bool = args.pop("export_controlnet")
export_diffusion_model: bool = args.pop("export_diffusion_model")
export_autoencoder: bool = args.pop("export_autoencoder")
onnx_export: bool = export_controlnet or export_autoencoder

a_prompt = "best quality, extremely detailed"
n_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
num_samples = 1
image_resolution = 512
ddim_steps = 20
guess_mode = False
strength = 1
scale = 9
seed = 1057192516
eta = 0.0


def canny2image():
    from gradio_canny2image import process

    input_image = cv2.imread("test_imgs/bird.png")
    input_image = input_image[:, :, ::-1]
    prompt = "bird"
    low_threshold = 100
    high_threshold = 200

    imgs = process(
        input_image, prompt, a_prompt, n_prompt, num_samples,
        image_resolution, ddim_steps, guess_mode, strength, scale,
        seed, eta, low_threshold, high_threshold
    )

    return imgs


def cli():
    imgs = canny2image()

    imgs = np.concatenate(imgs, axis=1)
    cv2.imwrite("output.png", imgs[:, :, ::-1])  # RGB -> BGR


if __name__ == '__main__':
    cli()
