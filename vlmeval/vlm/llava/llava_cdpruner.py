import torch
from PIL import Image
from abc import abstractproperty
import sys
import os.path as osp
from ..base import BaseModel
from ...smp import *
from ...dataset import DATASET_TYPE, DATASET_MODALITY
import copy
import requests


class LLaVA_CDPruner(BaseModel):
    """
    LLaVA model with CDPruner's Conditional DPP token pruning.
    Supports configurable visual_token_num for pruning.
    """

    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self, model_path="liuhaotian/llava-v1.5-7b", visual_token_num=64, **kwargs):
        """
        Initialize LLaVA with CDPruner token pruning.
        
        Args:
            model_path: Path to LLaVA model checkpoint
            visual_token_num: Number of visual tokens to keep after pruning (default: 64)
            **kwargs: Additional generation kwargs
        """
        try:
            from vlmeval.llava_cdpruner.model.builder import load_pretrained_model
        except Exception as err:
            logging.critical(
                "Failed to import CDPruner local LLaVA package. "
                "Please ensure vlmeval/llava_cdpruner/ exists with all required files."
            )
            raise err

        assert osp.exists(model_path) or splitlen(model_path) == 2
        self.visual_token_num = visual_token_num
        self.system_prompt = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions. "
        )
        self.stop_str = "</s>"

        # For LLaVA-1.5, always use the llava-v1.5 model name
        model_name = "llava-v1.5-7b"

        try:
            self.tokenizer, self.model, self.image_processor, self.context_len = (
                load_pretrained_model(
                    model_path=model_path,
                    model_base=None,
                    model_name=model_name,
                    device_map="cpu",
                    visual_token_num=visual_token_num,
                )
            )
        except Exception as err:
            logging.critical(f"Error loading LLaVA model with CDPruner: {err}")
            raise err

        self.model = self.model.cuda()
        self.conv_mode = "llava_v1"

        kwargs_default = dict(
            do_sample=False,
            temperature=0,
            max_new_tokens=2048,
            top_p=None,
            num_beams=1,
            use_cache=True,
        )
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        
        logging.info(
            f"LLaVA_CDPruner initialized with visual_token_num={visual_token_num}. "
            f"Generation config: {self.kwargs}"
        )

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == "MCQ":
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line["question"]
        hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
        if hint is not None:
            question = hint + "\n" + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f"\n{key}. {item}"
        prompt = question

        if len(options):
            prompt += (
                "\n请直接回答选项字母。"
                if cn_string(prompt)
                else "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += (
                "\n请直接回答问题。"
                if cn_string(prompt)
                else "\nAnswer the question directly."
            )

        message = [dict(type="image", value=s) for s in tgt_path]
        message.append(dict(type="text", value=prompt))
        return message

    def concat_tilist(self, message):
        text, images = "", []
        for item in message:
            if item["type"] == "text":
                text += item["value"]
            elif item["type"] == "image":
                text += " <image> "
                images.append(item["value"])
        return text, images

    def chat_inner(self, message, dataset=None):
        from vlmeval.llava_cdpruner.mm_utils import (
            process_images,
            tokenizer_image_token,
            KeywordsStoppingCriteria,
        )
        from vlmeval.llava_cdpruner.constants import IMAGE_TOKEN_INDEX

        prompt = self.system_prompt
        images = []
        for utter in message:
            prompt += "USER: " if utter["role"] == "user" else "ASSISTANT: "
            content, images_sub = self.concat_tilist(utter["content"])
            prompt += content
            images.extend(images_sub)
            prompt += " " if utter["role"] == "user" else self.stop_str
        assert message[-1]["role"] == "user", message
        prompt += "ASSISTANT: "

        images = [Image.open(s).convert("RGB") for s in images]
        args = abstractproperty()
        args.image_aspect_ratio = "pad"
        image_tensor = process_images(images, self.image_processor, args).to(
            "cuda", dtype=torch.float16
        )

        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )
        keywords = [self.stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )
        
        # CDPruner: Pass texts for token pruning
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                texts=prompt,  # CDPruner modification
                stopping_criteria=[stopping_criteria],
                **self.kwargs,
            )
        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()
        return output

    def generate_inner(self, message, dataset=None):
        from vlmeval.llava_cdpruner.mm_utils import (
            process_images,
            tokenizer_image_token,
            KeywordsStoppingCriteria,
        )
        from vlmeval.llava_cdpruner.constants import IMAGE_TOKEN_INDEX

        # Support interleave text and image
        content, images = self.concat_tilist(message)

        images = [Image.open(s).convert("RGB") for s in images]
        args = abstractproperty()
        args.image_aspect_ratio = "pad"
        if images:
            image_tensor = process_images(images, self.image_processor, args).to(
                "cuda", dtype=torch.float16
            )
        else:
            image_tensor = None

        prompt = self.system_prompt + "USER: " + content + " ASSISTANT: "

        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )
        keywords = [self.stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )
        
        # CDPruner: Pass texts for token pruning
        with torch.inference_mode():
            # CDPruner's generate returns (output_ids, visual_token_num)
            output_ids, visual_token_num = self.model.generate(
                input_ids,
                images=image_tensor,
                texts=prompt,  # CDPruner modification
                stopping_criteria=[stopping_criteria],
                **self.kwargs,
            )

        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()
        return output

