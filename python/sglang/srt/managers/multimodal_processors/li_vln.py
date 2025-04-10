import asyncio
from typing import List, Union

from sglang.srt.managers.multimodal_processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
    get_global_processor,
)
from sglang.srt.models.livln import VideoLlavaQwenForCausalLM


class VLNImageProcessor(BaseMultimodalProcessor):
    models = [VideoLlavaQwenForCausalLM]

    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)
        self.IMAGE_TOKEN = "<image>"

    @staticmethod
    def _process_images_task(images, input_text):
        processor = get_global_processor()
        result = processor.__call__(
            prompt=input_text, images=images, return_tensors="pt"
        )
        return {
            "input_ids": result["input_ids"],
            "pixel_values": result["pixel_values"],
        }

    async def _process_images(self, images, input_text):
        if self.executor is not None:
            loop = asyncio.get_event_loop()
            image_inputs = await loop.run_in_executor(
                self.executor,
                VLNImageProcessor._process_images_task,
                images,
                input_text,
            )
        else:
            image_inputs = self._processor(
                images=images, text=input_text, return_tensors="pt"
            )

        return image_inputs

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_ids,
        request_obj,
        max_req_input_len,
        **kwargs,
    ):
        if not image_data:
            return None

        if not isinstance(image_data, list):
            image_data = [image_data]

        base_out = self.load_mm_data(
            input_ids=input_ids,
            image_data=image_data,
            multimodal_tokens=MultimodalSpecialTokens(
                image_token=self.IMAGE_TOKEN
            ),
            max_req_input_len=max_req_input_len,
        )
        images = base_out.images
        res = await self._process_images(images=images, input_text=base_out.input_text)
        return {
            "input_ids": res["input_ids"].flatten().tolist(),
            "pixel_values": res["pixel_values"],
            "data_hashes": base_out.mm_data_hashes,
            "image_sizes": base_out.image_sizes,
        }
