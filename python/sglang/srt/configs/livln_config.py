import numpy as np
import os
import torchvision
import torch

from dataclasses import dataclass
from typing import Dict
from PIL.Image import Image
from transformers import Qwen2Config, PretrainedConfig, AutoImageProcessor, AutoProcessor, ProcessorMixin
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import to_numpy_array

from sglang.srt.conversation import LiConversation, LiSeparatorStyle



from typing import (
    List,
    Tuple,
    Union,
)

from enum import auto, Enum


class VideoLlavaQwenConfig(Qwen2Config):
    model_type = "video_llava_qwen"


IMAGE_TOKEN_INDEX = -200

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    image_token = "<image>"
    if "<video>" in prompt:
        image_token = "<video>"
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(image_token)]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

class VLMImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        image_size: int,
        min_size: int = 14,
        image_mean: Union[Tuple[float, float, float], List[float]] = (
            0.48145466,
            0.4578275,
            0.40821073,
        ),
        image_std: Union[Tuple[float, float, float], List[float]] = (
            0.26862954,
            0.26130258,
            0.27577711,
        ),
        rescale_factor: float = 1.0 / 255.0,
        do_normalize: bool = True,
        do_rescale: bool = True,
        use_high_res_cam120_area: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        if isinstance(image_size, tuple) or isinstance(image_size, list):
            self.image_h = self.image_size[0]
            self.image_w = self.image_size[1]
        else:
            self.image_h = self.image_size
            self.image_w = self.image_size
        self.crop_size = dict()
        self.crop_size['width'] = self.image_w
        self.crop_size['height'] = self.image_h
        self.rescale_factor = rescale_factor
        self.image_mean = image_mean
        self.image_std = image_std
        self.min_size = min_size
        self.do_normalize = do_normalize
        self.do_rescale = do_rescale
        self.use_high_res_cam120_area = use_high_res_cam120_area
        self.need_crop_image = (os.getenv('NEED_CROP_IMAGE', "False") == "True")

        if image_mean is None:
            self.background_color = (127, 127, 127)
        else:
            self.background_color = tuple([int(x * 255) for x in image_mean])

    def resize(self, pil_img: Image) -> np.ndarray:
        width, height = pil_img.size

        max_size = max(width, height)

        size = [
            max(int(height / max_size * self.image_h), self.min_size),
            max(int(width / max_size * self.image_w), self.min_size),
        ]

        if width <= 0 or height <= 0 or size[0] <= 0 or size[1] <= 0:
            print(f"orig size = {pil_img.size}, new size = {size}")
            raise ValueError("Invalid size!")

        pil_img = torchvision.transforms.functional.resize(
            pil_img,
            size,
            interpolation=torchvision.transforms.functional.InterpolationMode.BILINEAR,
            # antialias=False,
        )

        x = to_numpy_array(pil_img)

        # [H, W, 3] -> [3, H, W]
        x = np.transpose(x, (2, 0, 1))

        return x

    def pad(self, np_img):
        h_padding = max(self.image_h - np_img.shape[1], 0)
        w_padding = max(self.image_w - np_img.shape[2], 0)
        top = h_padding // 2
        bottom = h_padding - top
        left = w_padding // 2
        right = w_padding - left
        new_image = np.pad(np_img, pad_width=((0, 0), (top, bottom), (left, right)), mode='constant', constant_values=0)

        return new_image

    def preprocess(self, images, return_tensors: str = "pt",
                   **kwargs) -> BatchFeature:
        # resize and pad to [self.image_size, self.image_size]
        # then convert from [H, W, 3] to [3, H, W]
        if type(images) is not list:
            images = [images]
        # if type(image_paths) is not list:
        #     image_paths = [image_paths]
        # assert len(images) == len(image_paths)
        # keywords = [
        # "dm-data", "dm-dataset", "dm-dataset-badcase",
        # "toll-lane", "vlm-lane-qa",
        # "/lpai/dataset/ad-prod-", "/lpai/dataset/ad-autopilot-",
        # "vlm-parking-navi-dataset", "train-occ-parking-occ",
        # "qa_result", "demo"
        # ]
        high_res_crop_image = []
        # for i in range(len(image_paths)):
        #     image = images[i]
        #     image_path = image_paths[i]
        #     if image_path and (self.need_crop_image or any(keyword in image_path for keyword in keywords)):
        #         width, height = image.size
        #         if self.image_h == 384 and self.image_w == 960:
        #             if width == 3840 and height == 2160:
        #                 crop_area = (0, 12, width, height - 612)
        #             elif width == 1920 and height == 1080:
        #                 crop_area = (0, 6, width, height - 306)
        #             crop_image = image.crop(crop_area)
        #             images[i] = crop_image

        #         if (self.image_h == 512 and self.image_w == 1280) or (self.image_h == 768 and self.image_w == 1920):
        #             if width == 3840 and height == 2160:
        #                 crop_area = (0, 12, width, height - 612)
        #             elif width == 1920 and height == 1080:
        #                 crop_area = (0, 6, width, height - 306)
        #             crop_area = (0, 6, width, height - 306)
        #             crop_image = image.crop(crop_area)
        #             images[i] = crop_image
        #         elif self.image_h == 336 and self.image_w == 840:
        #             if width == 3840 and height == 2160:
        #                 crop_area = (240, 200, width - 240, height - 616)
        #             elif width == 1920 and height == 1080:
        #                 crop_area = (120, 100, width - 120, height - 308)
        #             crop_image = image.crop(crop_area)
        #             images[i] = crop_image

        images: List[np.ndarray] = [self.resize(image) for image in images]

        # LC: for high resolution crop
        if high_res_crop_image:
            high_res_crop_image: List[np.ndarray] = [self.resize(image) for image in high_res_crop_image]

        # resacle from [0, 255] -> [0, 1]
        if self.do_rescale:
            images = [
                self.rescale(
                    image=image,
                    scale=self.rescale_factor,
                    input_data_format="channels_first",
                )
                for image in images
            ]

            # LC: for high resolution crop
            if high_res_crop_image:
                high_res_crop_image = [
                    self.rescale(
                        image=image,
                        scale=self.rescale_factor,
                        input_data_format="channels_first",
                    )
                    for image in high_res_crop_image
                ]

        # normalize
        if self.do_normalize:
            images = [
                self.normalize(
                    image=image,
                    mean=self.image_mean,
                    std=self.image_std,
                    input_data_format="channels_first",
                )
                for image in images
            ]

            # LC: for high resolution crop
            if high_res_crop_image:
                high_res_crop_image = [
                    self.normalize(
                        image=image,
                        mean=self.image_mean,
                        std=self.image_std,
                        input_data_format="channels_first",
                    )
                    for image in high_res_crop_image
                ]

        images = [self.pad(image) for image in images]
        
        # LC: for high resolution crop
        if high_res_crop_image:
            high_res_crop_image = [self.pad(image) for image in high_res_crop_image]

        data = {"pixel_values": images}

        # LC: for high resolution crop
        if high_res_crop_image:
            crop_data = {"pixel_values": high_res_crop_image}
            assert False, "do not support high res crop image"
            return BatchFeature(data=data, tensor_type=return_tensors), BatchFeature(data=crop_data, tensor_type=return_tensors)
        return BatchFeature(data=data, tensor_type=return_tensors)

    @property
    def default_shape(self):
        return [3, self.image_h, self.image_w]


class SigLIPImageProcessor(VLMImageProcessor):
    model_input_names = ["pixel_values"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image_mean = [0.5, 0.5, 0.5]
        self.image_std = [0.5, 0.5, 0.5]
        self.background_color = tuple([int(x * 255) for x in self.image_mean])

    def resize(self, pil_img: Image) -> np.ndarray:
        width, height = pil_img.size
        max_ratio = min(self.image_h / height, self.image_w / width)
        size = [int(max_ratio * height), int(max_ratio * width)]
        if width <= 0 or height <= 0 or size[0] <= 0 or size[1] <= 0:
            print(f"orig size = {pil_img.size}, new size = {size}")
            raise ValueError("Invalid size!")
        pil_img = torchvision.transforms.functional.resize(
            pil_img,
            size,
            interpolation=torchvision.transforms.functional.InterpolationMode.BILINEAR,
            # antialias=False,
        )

        x = to_numpy_array(pil_img)
        x = np.transpose(x, (2, 0, 1))
        return x


class VLMImageProcessorConfig(PretrainedConfig):
    model_input_names = ["pixel_values"]
    model_type = "video_llava_qwen"
    image_size: int
    min_size: int
    image_mean: Union[Tuple[float, float, float], List[float]]
    image_std: Union[Tuple[float, float, float], List[float]]
    rescale_factor: float
    do_normalize: bool
    
    use_high_res_cam120_area: bool = False
    do_rescale: bool = True

    def __init__(
        self,
        image_size: int,
        min_size: int = 14,
        image_mean: Union[Tuple[float, float, float], List[float]] = (
            0.48145466,
            0.4578275,
            0.40821073,
        ),
        image_std: Union[Tuple[float, float, float], List[float]] = (
            0.26862954,
            0.26130258,
            0.27577711,
        ),
        rescale_factor: float = 1.0 / 255.0,
        do_normalize: bool = True,
        do_rescale: bool = True,
        use_high_res_cam120_area: bool = False,
        **kwargs,
    ):
        self.image_size = image_size
        self.min_size = min_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize

        super().__init__(**kwargs)


class VLNProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    # valid_kwargs = ["chat_template"]

    tokenizer_class = ("GPT2Tokenizer", "GPT2TokenizerFast")
    image_processor_class = "SigLIPImageProcessor"
    
    def __init__(
        self,
        image_processor= None,
        tokenizer = None,
        chat_template = None,
        num_image_tokens: int = 361,
        add_special_token: bool = False,
        **kwargs,
    ):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.conv_mode = LiConversation(
                name="li-vln",
                system="",
                roles=("USER", "ASSISTANT"),
                version="llama_v2",
                messages=(),
                offset=0,
                sep_style=LiSeparatorStyle.LLAMA_2,
                sep="",
                sep2="<|endoftext|>",
                image_token="<|image|>",
            )
        # self.image_tag = "<image>"
        self.image_token_index = IMAGE_TOKEN_INDEX
        self.num_image_tokens = num_image_tokens
        self.add_special_token = add_special_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token_id = 0
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        names_from_processor = list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
        return names_from_processor + ["second_per_grid_ts"]
    
    # @property
    # def image_id(self) -> int:
    #     image_id = self.tokenizer.vocab.get(self.image_tag)
    #     return image_id

    def add_image_token(
        self,
        image_indices: List[int],
        input_ids: torch.LongTensor,
    ):
        """

        Args:
            # image_indices (List[int]): [index_0, index_1, ..., index_j]
            input_ids (torch.LongTensor): [N]

        Returns:
            input_ids (torch.LongTensor): [N + image tokens]
            num_image_tokens (torch.IntTensor): [n_images]
        """
        input_slices = []

        start = 0
        for index in image_indices:
            if self.add_special_token:
                end = index + 1
            else:
                end = index

            # original text tokens
            input_slices.append(input_ids[start:end])
           
            input_slices.append(
                self.image_token_index * torch.ones((self.num_image_tokens,), dtype=torch.long)
            )
            start = index + 1

        # the left part
        input_slices.append(input_ids[start:])
        # concat all slices
        input_ids = torch.cat(input_slices, dim=0)
    
        num_image_tokens = torch.IntTensor([self.num_image_tokens] * len(image_indices))
        return input_ids, num_image_tokens

    def __call__(
        self,
        *,
        prompt: str = None,
        conversations: List[Dict[str, str]] = None,
        images: List[Image] = None,
        return_tensors: str = "pt",
        **kwargs,
    ) -> BatchFeature:
        # process images
        new_images = []
        for image in images:         
            new_image = self.image_processor.preprocess([image], return_tensors='pt')['pixel_values'][0]
            new_images.append(new_image)
        image_inputs = torch.stack(new_images, dim=0)[0]
        
        # we have add <image> in the end of prompt, remove it
        if prompt.endswith("<image>"):
            prompt = prompt[:-7].strip() + " "
        qs = "<image>" + '\n' + prompt
        conv = self.conv_mode.copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt, self.tokenizer, self.image_token_index, return_tensors='pt')

        # add image tokens to the input_ids
        image_token_mask: torch.Tensor = (input_ids == self.image_token_index).to(torch.bool)
        image_indices = image_token_mask.nonzero()
        input_ids, num_image_tokens = self.add_image_token(
            image_indices,
            input_ids,
        )
        data = {
            "pixel_values": image_inputs,
            "input_ids": input_ids,
            "videos_inputs": None,
            "num_image_tokens": num_image_tokens,
        }

        return BatchFeature(data=data)

VLNProcessor.register_for_auto_class('AutoProcessor')
SigLIPImageProcessor.register_for_auto_class('AutoImageProcessor')

AutoProcessor.register(VLMImageProcessorConfig , VLNProcessor)
AutoImageProcessor.register(VLMImageProcessorConfig , SigLIPImageProcessor)