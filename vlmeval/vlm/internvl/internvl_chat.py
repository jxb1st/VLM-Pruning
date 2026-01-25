import math
import pandas as pd
import random
import re
import yaml
import string
import torch
import torch.distributed as dist
import torchvision.transforms as T
import transformers
import warnings
from PIL import Image
from functools import partial
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoConfig, AutoModel, CLIPImageProcessor

from .utils import (build_multi_choice_prompt,
                    build_video_prompt,
                    build_mpo_prompt,
                    build_mcq_cot_prompt,
                    build_qa_cot_prompt,
                    mpo_post_processing,
                    format_nav_prompt,
                    pile_action_history,
                    reorganize_prompt,
                    load_image)
from .utils import mpo_prompt_with_final_answer, mpo_prompt_without_final_answer, parse_bbox_internvl

from ..base import BaseModel
from ...dataset import DATASET_TYPE, DATASET_MODALITY, build_dataset, infer_dataset_basename
from ...smp import *

# load all the gui templates
upper_path = Path(__file__).parent
with open(os.path.join(upper_path, "gui_template.yaml"), "r") as f:
    GUI_TEMPLATE = yaml.load(f, Loader=yaml.FullLoader)

R1_SYSTEM_PROMPT = """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different \
angles, potential solutions, and reason through the problem step-by-step. \
Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to \
the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the \
query. The final answer should be standalone and not reference the thinking \
section.
""".strip()


def prepare_messages_list(prompt, image_path, system_prompt=None):
    from lmdeploy.vl.constants import IMAGE_TOKEN
    content = [{'type': 'text', 'text': prompt.replace('<image>', IMAGE_TOKEN)}]

    if isinstance(image_path, str):
        image_path = [image_path]

    for image in image_path:
        img = Image.open(image).convert('RGB')
        b64 = encode_image_to_base64(img)
        img_struct = dict(url=f'data:image/jpeg;base64,{b64}')
        content.append(dict(type='image_url', image_url=img_struct))

    messages = []

    if system_prompt is not None:
        messages.append({'role': 'system', 'content': system_prompt})

    messages.append({
        'role': 'user',
        'content': content,
    })

    return [messages]


def extract_boxed_content(ans: str):
    idx = ans.rfind(r'\boxed{')
    if idx == -1:
        return ans

    idx += len(r'\boxed{')
    brace_level = 1
    content_start = idx
    i = idx

    while i < len(ans):
        if ans[i] == '{':
            brace_level += 1
        elif ans[i] == '}':
            brace_level -= 1
            if brace_level == 0:
                break
        i += 1

    if brace_level != 0:
        # Unbalanced braces
        return ans

    content = ans[content_start:i]
    return content


class InternVLChat(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self,
                 model_path='OpenGVLab/InternVL-Chat-V1-5',
                 load_in_8bit=False,
                 use_mpo_prompt=False,
                 version='V1.0',
                 screen_parse=True,
                 # Best-of-N parameters
                 best_of_n=1,
                 reward_model_path=None,
                 # R1 parameters
                 cot_prompt_version='v1',
                 #
                 use_lmdeploy=False,
                 use_postprocess=False,
                 **kwargs):

        assert best_of_n >= 1
        assert model_path is not None
        assert version_cmp(transformers.__version__, '4.37.2', 'ge')

        self.use_lmdeploy = use_lmdeploy
        self.cot_prompt_version = cot_prompt_version
        self.use_mpo_prompt = use_mpo_prompt
        self.use_cot = (os.getenv('USE_COT') == '1')
        self.use_postprocess = use_postprocess

        if cot_prompt_version == 'r1':
            self.system_prompt = R1_SYSTEM_PROMPT
            self.cot_prompt = 'Please answer the question and put the final answer within \\boxed{}.'
        elif cot_prompt_version == 'v2':
            self.system_prompt = None
            self.cot_prompt = "Answer the preceding multiple-choice question \
            by carefully analyzing the provided image. \nPlease answer with \
            carefully thought step by step. Apply the thinking process \
            recursively at both macro and micro levels. \nVerify consistency \
            of reasoning and look for potential flaws or gaps during \
            thinking. \nWhen realize mistakes, explain why the previous \
            thinking was incorrect, fix it and then continue thinking.\nThe \
            last line of your response should follow this format: 'Answer: \
            \\boxed{$LETTER}' (without quotes), where LETTER is one of the \
            options\n\n"
        else:
            assert cot_prompt_version == 'v1'
            self.system_prompt = None
            self.cot_prompt = None

        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

        # Regular expression to match the pattern 'Image' followed by a number, e.g. Image1
        self.pattern = r'Image(\d+)'
        # Replacement pattern to insert a hyphen between 'Image' and the number, e.g. Image-1
        self.replacement = r'Image-\1'

        # Convert InternVL2 response to dataset format
        # e.g. Image1 -> Image-1

        # Regular expression to match the pattern 'Image-' followed by a number
        self.reverse_pattern = r'Image-(\d+)'
        # Replacement pattern to remove the hyphen (Image-1 -> Image1)
        self.reverse_replacement = r'Image\1'

        self.screen_parse = screen_parse

        if use_lmdeploy:
            from lmdeploy import TurbomindEngineConfig, PytorchEngineConfig, VisionConfig, pipeline
            engine_type = PytorchEngineConfig if "internvl3_5" in model_path.lower() else TurbomindEngineConfig
            vision_config = VisionConfig(max_batch_size=4)
            num_gpus = torch.cuda.device_count()
            self.model = pipeline(
                model_path,
                vision_config=vision_config,
                backend_config=engine_type(
                    session_len=max(16384, kwargs.get("max_new_tokens", 16384)),
                    cache_max_entry_count=0.5,
                    tp=num_gpus,
                )
            )
            torch.cuda.set_device(0)
            self.device = 'cuda'
        else:
            # Try to use local model first (for InternVL3.5)
            local_model_loaded = False
            if "internvl3" in model_path.lower() and ("3_5" in model_path.lower() or "3.5" in model_path.lower()):
                try:
                    # Determine which local model to use
                    if "8b" in model_path.lower():
                        from .local_models.internvl3_5_8b.modeling_internvl_chat import InternVLChatModel as LocalModel
                        from .local_models.internvl3_5_8b.configuration_internvl_chat import InternVLChatConfig as LocalConfig
                        local_model_dir = "internvl3_5_8b"
                    elif "30b" in model_path.lower() and ("a3b" in model_path.lower() or "moe" in model_path.lower()):
                        from .local_models.internvl3_5_30b_a3b.modeling_internvl_chat import InternVLChatModel as LocalModel
                        from .local_models.internvl3_5_30b_a3b.configuration_internvl_chat import InternVLChatConfig as LocalConfig
                        local_model_dir = "internvl3_5_30b_a3b"
                    else:
                        raise ImportError("Unknown InternVL3.5 variant")
                    
                    # Load local config.json if it exists
                    local_config_dir = Path(__file__).parent / "local_models" / local_model_dir
                    local_config_path = local_config_dir / "config.json"
                    
                    if local_config_path.exists():
                        print(f"[INFO] Loading local config from: {local_config_path}")
                        config = LocalConfig.from_pretrained(str(local_config_dir))
                    else:
                        print(f"[INFO] Local config not found, loading from HuggingFace cache")
                        config = LocalConfig.from_pretrained(model_path)
                    
                    # Load using local model code with config
                    self.model = LocalModel.from_pretrained(
                        model_path,
                        config=config,
                        torch_dtype=torch.bfloat16,
                        load_in_8bit=load_in_8bit,
                        low_cpu_mem_usage=True,
                        device_map="auto").eval()
                    
                    self.device = 'cuda'
                    local_model_loaded = True
                    print(f"[INFO] Loaded InternVL3.5 using local model code: {local_model_dir}")
                    print(f"[INFO] Tile Pruning Config: enable={config.enable_tile_pruning}, keep_ratio={config.tile_keep_ratio}")
                except (ImportError, Exception) as e:
                    print(f"[WARNING] Failed to load local model, falling back to HuggingFace: {e}")
            
            if not local_model_loaded:
                # Fallback to HuggingFace with trust_remote_code
                self.model = AutoModel.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    load_in_8bit=load_in_8bit,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    device_map="auto").eval()
                print("[INFO] Loaded model using HuggingFace AutoModel with trust_remote_code")
                
                self.device = 'cuda'

        if best_of_n > 1:
            assert version == 'V2.0', 'only support BoN evaluation with version==V2.0'
            assert reward_model_path is not None

            self.reward_tokenizer = AutoTokenizer.from_pretrained(
                reward_model_path, trust_remote_code=True, use_fast=False)
            self.reward_model = AutoModel.from_pretrained(
                reward_model_path,
                torch_dtype=torch.bfloat16,
                load_in_8bit=load_in_8bit,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map="auto").eval()

            if not self.use_cot:
                os.environ['USE_COT'] = '1'
                self.use_cot = True
                print('[Warning] Since Best-of-N is enabled, USE_COT is forced to be set to 1.')

            print(f'Enable Best-of-N evaluation with PRM: {reward_model_path}')

        # self.image_size = self.model.config.vision_config.image_size
        self.version = version
        self.best_of_n = best_of_n
        kwargs_default = dict(do_sample=False, max_new_tokens=4096, top_p=None)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default

        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')
        
        # [DEBUG] Initialize debug logger for token pruning research
        from .debug_logger import InternVLDebugLogger
        from . import utils
        self.debug_logger = InternVLDebugLogger()
        utils.set_debug_logger(self.debug_logger)
        
        # [DEBUG] Inject debug logger into local model if using local model code
        if not use_lmdeploy and hasattr(self, 'model'):
            try:
                from .local_models.debug_utils import set_debug_logger
                set_debug_logger(self.debug_logger)
                print("[DEBUG] Debug logger injected into local InternVL model")
            except ImportError:
                # If local_models module doesn't exist or model loaded via HuggingFace, 
                # debug logging won't be available inside the model
                pass

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if dataset in [
            'atomic_dataset', 'electro_dataset', 'mechanics_dataset',
            'optics_dataset', 'quantum_dataset', 'statistics_dataset'
        ]:
            return False
        if listinstr(['MMDU', 'MME-RealWorld', 'MME-RealWorld-CN', 'WeMath_COT', 'MMAlignBench', 'ChartQAPro', 'ChartMuseum', 'MMSIVideo_U50','MMSIVideo_SC'], dataset):  # noqa: E501
            # For Multi-Turn we don't have custom prompt
            return False
        if DATASET_TYPE(dataset) == 'MCQ':
            if dataset is not None and 'LEGO' in dataset:
                return False
        if DATASET_MODALITY(dataset) == 'VIDEO':
            # For Video benchmarks we don't have custom prompt at here
            return False
        else:
            return True

    def build_prompt(self, line, dataset=None):
        use_mpo_prompt = self.use_mpo_prompt and (self.use_cot or dataset in ['MMStar', 'HallusionBench', 'OCRBench'])

        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)
        if dataset is not None and listinstr(['BMMR'], dataset):
            self.kwargs['max_new_tokens'] = max(self.kwargs.get('max_new_tokens', 4096), 8196)
            print(f'[Warning] BMMR dataset requires a larger max_new_tokens, set to {self.kwargs["max_new_tokens"]}')

        if dataset is not None and DATASET_TYPE(dataset) == 'Y/N':
            question = line['question']
            if listinstr(['MME'], dataset):
                prompt = question + ' Answer the question using a single word or phrase.'
            elif listinstr(['HallusionBench', 'AMBER'], dataset):
                prompt = question + ' Please answer yes or no. Answer the question using a single word or phrase.'
            else:
                prompt = question
        elif dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
            prompt = build_multi_choice_prompt(line, dataset)
            if os.getenv('USE_COT') == '1':
                prompt = build_mcq_cot_prompt(line, prompt, self.cot_prompt)
        elif dataset is not None and DATASET_TYPE(dataset) == 'VQA':
            question = line['question']
            if listinstr(['LLaVABench', 'WildVision'], dataset):
                prompt = question + '\nAnswer this question in detail.'
            elif listinstr(['OCRVQA', 'TextVQA', 'ChartQA', 'DocVQA', 'InfoVQA', 'OCRBench',
                            'DUDE', 'SLIDEVQA', 'GQA', 'MMLongBench_DOC'], dataset):
                prompt = question + '\nAnswer the question using a single word or phrase.'
            elif listinstr(['MathVerse'], dataset):
                question = question.replace("please directly answer the question and", "please")
                prompt = question
                if os.getenv('USE_COT') == '1':
                    prompt = build_qa_cot_prompt(line, prompt, self.cot_prompt)
            elif listinstr(['MathVista', 'MathVision', 'VCR', 'MTVQA', 'MMVet',
                            'MMDU', 'CRPE', 'MIA-Bench', 'MM-Math', 'DynaMath', 'QSpatial',
                            'WeMath', 'LogicVista', 'MM-IFEval', 'ChartMimic', 'MMReason'], dataset):
                prompt = question
                if os.getenv('USE_COT') == '1':
                    prompt = build_qa_cot_prompt(line, prompt, self.cot_prompt)
            else:
                prompt = question + '\nAnswer the question using a single word or phrase.'
        elif dataset is not None and DATASET_TYPE(dataset) == 'GUI':
            ds_basename = infer_dataset_basename(dataset)
            ds = build_dataset(dataset, skeleton=True)
            action_space = ds.get_action_space()
            traj_dict = ds.get_trajectory(line)

            prompt_config = GUI_TEMPLATE[ds_basename]
            if 'history' in prompt_config["placeholders"]:
                traj_dict['history'] = pile_action_history(traj_dict['history'])
            prompt = format_nav_prompt(
                (
                    "Please provide the bounding box coordinate of the region this sentence describes: <ref>{task}</ref>"  # noqa: E501
                    if self.screen_parse
                    else prompt_config["template"]
                ),
                prompt_config["placeholders"],
                action_space=action_space,
                **traj_dict,
            )
        else:
            # VQA_ex_prompt: OlympiadBench, VizWiz
            prompt = line['question']
            if os.getenv('USE_COT') == '1':
                prompt = build_qa_cot_prompt(line, prompt, self.cot_prompt)

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])

        if use_mpo_prompt:
            message = build_mpo_prompt(message, line, dataset)
        return message

    def set_max_num(self, dataset):
        # The total limit on the number of images processed, set to avoid Out-of-Memory issues.
        self.total_max_num = 64
        if dataset is None:
            self.max_num = 6
            return None
        res_12_datasets = ['ChartQA_TEST', 'MMMU_DEV_VAL', 'MMMU_TEST', 'MME-RealWorld',
                           'VCR_EN', 'VCR_ZH', 'OCRVQA', 'BMMR']
        res_18_datasets = ['DocVQA_VAL', 'DocVQA_TEST', 'DUDE', 'MMLongBench_DOC', 'SLIDEVQA']
        res_24_datasets = ['InfoVQA_VAL', 'InfoVQA_TEST', 'OCRBench', 'HRBench4K', 'HRBench8K']
        if DATASET_MODALITY(dataset) == 'VIDEO':
            self.max_num = 1
        elif listinstr(res_12_datasets, dataset):
            self.max_num = 12
        elif listinstr(res_18_datasets, dataset):
            self.max_num = 18
        elif listinstr(res_24_datasets, dataset):
            self.max_num = 24
        elif DATASET_TYPE(dataset) == 'GUI':
            self.max_num = 12
        else:
            self.max_num = 6

    def generate_v1_2(self, message, dataset=None):
        self.INTERLEAVE = False
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.image_size, self.image_size))
        image_processor = CLIPImageProcessor.from_pretrained(self.model_path)
        pixel_values = image_processor(images=image, return_tensors='pt').pixel_values
        pixel_values = pixel_values.to(torch.bfloat16).to(self.device)
        with torch.no_grad():
            response = self.model.chat(self.tokenizer, pixel_values=pixel_values,
                                       question=prompt, generation_config=self.kwargs)
        return response

    def generate_v1_5(self, message, dataset=None):
        image_num = len([x for x in message if x['type'] == 'image'])
        max_num = max(1, min(self.max_num, self.total_max_num // image_num))
        prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])

        if DATASET_MODALITY(dataset) == 'VIDEO':
            prompt = build_video_prompt(prompt, dataset)

        if image_num > 1:
            image_path = [x['value'] for x in message if x['type'] == 'image']
            pixel_values_list = []
            for file_name in image_path:
                pixel_values_list.append(load_image(file_name, max_num=max_num).to(self.device).to(torch.bfloat16))
            pixel_values = torch.cat(pixel_values_list, dim=0)
        elif image_num == 1:
            image_path = [x['value'] for x in message if x['type'] == 'image'][0]
            pixel_values = load_image(image_path, max_num=max_num).to(self.device).to(torch.bfloat16)
        else:
            pixel_values = None
        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer,
                pixel_values=pixel_values,
                question=prompt,
                generation_config=self.kwargs,
                verbose=True)
        return response

    @torch.no_grad()
    def generate_v2(self, message, dataset=None):

        use_mpo_prompt = self.use_mpo_prompt and (self.use_cot or dataset in ['MMStar', 'HallusionBench', 'OCRBench'])

        image_num = len([x for x in message if x['type'] == 'image'])
        max_num = max(1, min(self.max_num, self.total_max_num // image_num))
        prompt = reorganize_prompt(message, image_num, dataset=dataset)
        
        # [DEBUG] Log input message information
        if self.debug_logger.should_log():
            text_parts = [x['value'] for x in message if x['type'] == 'text']
            text_content = ' '.join(text_parts)
            self.debug_logger.log_stage("2. Token Generation - Input", {
                "dataset": dataset,
                "image_num": image_num,
                "max_num": max_num,
                "total_max_num": self.total_max_num,
                "text_length": len(text_content),
                "prompt_length": len(prompt),
                "use_mpo_prompt": use_mpo_prompt,
                "text_preview": text_content[:200] + "..." if len(text_content) > 200 else text_content,
                "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt
            })

        if dataset is not None and DATASET_MODALITY(dataset) == 'VIDEO':
            prompt = build_video_prompt(prompt, dataset)

        if image_num > 1:
            image_path = [x['value'] for x in message if x['type'] == 'image']
            num_patches_list, pixel_values_list, grid_ratios = [], [], []
            for image_idx, file_name in enumerate(image_path):
                upscale_flag = image_idx == 0 and dataset is not None and listinstr(['MMMU'], dataset)
                curr_pixel_values, curr_grid_ratio = load_image(
                    file_name, max_num=max_num, upscale=upscale_flag)
                curr_pixel_values = curr_pixel_values.to(self.device).to(torch.bfloat16)
                num_patches_list.append(curr_pixel_values.size(0))
                pixel_values_list.append(curr_pixel_values)
                grid_ratios.append(curr_grid_ratio)
            pixel_values = torch.cat(pixel_values_list, dim=0)
            grid_ratio = grid_ratios[0] if len(grid_ratios) > 0 else None
        elif image_num == 1:
            image_path = [x['value'] for x in message if x['type'] == 'image'][0]
            upscale_flag = dataset is not None and listinstr(['MMMU'], dataset)
            pixel_values, grid_ratio = load_image(
                image_path, max_num=max_num, upscale=upscale_flag)
            pixel_values = pixel_values.to(self.device).to(torch.bfloat16)
            num_patches_list = [pixel_values.size(0)]
        else:
            pixel_values = None
            num_patches_list = []
            grid_ratio = None
        
        # [DEBUG] Log loaded image information
        if self.debug_logger.should_log() and pixel_values is not None:
            self.debug_logger.log_stage("2. Token Generation - Image Loaded", {
                "pixel_values_shape": str(tuple(pixel_values.shape)),
                "num_patches_list": str(num_patches_list),
                "total_patches": sum(num_patches_list),
                "patches_per_image": f"{sum(num_patches_list) / image_num:.2f}" if image_num > 0 else "N/A",
                "memory_mb": f"{pixel_values.element_size() * pixel_values.nelement() / 1024 / 1024:.2f}",
                "device": str(pixel_values.device),
                "dtype": str(pixel_values.dtype)
            })

        response_list = []
        for idx in range(self.best_of_n):
            kwargs_default = self.kwargs.copy()
            kwargs_default['do_sample'] = idx > 0 or kwargs_default.get('do_sample', False)
            kwargs_default['temperature'] = 0.6
            kwargs_default['top_p'] = 0.95

            if self.use_lmdeploy:
                from lmdeploy import GenerationConfig
                gen_config = GenerationConfig(**kwargs_default)
                gen_config.random_seed = None
                messages_list = prepare_messages_list(prompt, image_path, system_prompt=self.system_prompt)
                assert len(messages_list) == 1
                response = self.model(messages_list, gen_config=gen_config)[0]
                response = response.text
            else:
                # [DEBUG] Pass debug logger to HuggingFace model
                if not hasattr(self.model, 'debug_logger'):
                    self.model.debug_logger = self.debug_logger
                
                # [DEBUG] Pass dataset name for token stats
                if hasattr(self.model, 'debug_logger'):
                    self.model.debug_logger.current_dataset = dataset
                
                if self.system_prompt is not None:
                    self.model.system_message = self.system_prompt
                response = self.model.chat(
                    self.tokenizer,
                    pixel_values=pixel_values,
                    num_patches_list=num_patches_list,
                    question=prompt,
                    generation_config=kwargs_default,
                    verbose=idx == 0,
                    grid_ratio=grid_ratio if 'grid_ratio' in locals() else None,
                )
            response_list.append(response)
            
            # [DEBUG] Increment sample counter after first inference
            if idx == 0:
                self.debug_logger.increment_sample()

        if self.best_of_n > 1:
            response_list = self.reward_model.select_best_response(
                tokenizer=self.reward_tokenizer,
                question=prompt,
                response_list=response_list,
                pixel_values=pixel_values,
                num_patches_list=num_patches_list,
            )
        response = response_list[0]

        if dataset is not None and not listinstr(['WeMath'], dataset):
            if use_mpo_prompt:
                response = mpo_post_processing(response, dataset)
            elif self.use_cot and self.use_postprocess:
                response = extract_boxed_content(response)

        if dataset is not None and DATASET_TYPE(dataset) == 'GUI' and self.screen_parse:
            # Parse the bounding box coordinates from the response
            response = parse_bbox_internvl(response)
            # Normalize the coordinates to the range [0, 1]
            if isinstance(response, list):
                response = [item / 1000 for item in response]
                # Convert the coordinates to the format required by the GUI
                response = f"x={response[0]}, y={response[1]}"

        return response

    def generate_inner(self, message, dataset=None):
        self.set_max_num(dataset)
        # Removed: print(f'InternVL model version: {self.version}')
        if self.version in ['V1.1', 'V1.2']:
            return self.generate_v1_2(message, dataset)
        elif self.version == 'V1.5':
            return self.generate_v1_5(message, dataset)
        elif self.version == 'V2.0':
            return self.generate_v2(message, dataset)
        else:
            raise ValueError(f'Unsupported version: {self.version}')

    def build_history(self, message):
        # Global Variables
        image_path = []
        image_cnt = 0

        def concat_tilist(tilist):
            nonlocal image_cnt  # Declare image_cnt as nonlocal to modify it
            prompt = ''
            for item in tilist:
                # Substitute the pattern in the text
                if item['type'] == 'text':
                    prompt += re.sub(self.pattern, self.replacement, item['value'])
                elif item['type'] == 'image':
                    image_cnt += 1
                    prompt += '<image>\n'
                    image_path.append(item['value'])
            return prompt

        # Only previous messages
        assert len(message) % 2 == 0
        history = []
        for i in range(len(message) // 2):
            m1, m2 = message[2 * i], message[2 * i + 1]
            assert m1['role'] == 'user' and m2['role'] == 'assistant'
            history.append((concat_tilist(m1['content']), concat_tilist(m2['content'])))

        return history, image_path, image_cnt

    def chat_inner_v2(self, message, dataset=None):

        if len(message) > 1:
            history, image_path, image_cnt = self.build_history(message[:-1])
        else:
            history, image_path, image_cnt = None, [], 1
        current_msg = message[-1]
        question = ''

        # If message is just text in the conversation
        if len(current_msg['content']) == 1 and current_msg['content'][0]['type'] == 'text':
            question = current_msg['content'][0]['value']
            question = re.sub(self.pattern, self.replacement, question)  # Fix pattern as per InternVL
        else:
            for msg in current_msg['content']:
                if msg['type'] == 'text':
                    question += re.sub(self.pattern, self.replacement, msg['value'])
                elif msg['type'] == 'image':
                    image_cnt += 1
                    question += '<image>\n'
                    image_path.append(msg['value'])

        if image_cnt > 1:
            num_patches_list = []
            pixel_values_list = []
            grid_ratios = []
            for image_idx, file_name in enumerate(image_path):
                upscale_flag = image_idx == 0 and dataset is not None and listinstr(['MMMU_DEV_VAL'], dataset)
                curr_pixel_values, curr_grid_ratio = load_image(
                    file_name, max_num=self.max_num, upscale=upscale_flag)
                curr_pixel_values = curr_pixel_values.to(self.device).to(torch.bfloat16)
                num_patches_list.append(curr_pixel_values.size(0))
                pixel_values_list.append(curr_pixel_values)
                grid_ratios.append(curr_grid_ratio)
            pixel_values = torch.cat(pixel_values_list, dim=0)
            grid_ratio = grid_ratios[0] if len(grid_ratios) > 0 else None  # Use first image's grid
        elif image_cnt == 1:
            upscale_flag = listinstr(['MMMU_DEV_VAL'], dataset)
            pixel_values, grid_ratio = load_image(
                image_path, max_num=self.max_num, upscale=upscale_flag)
            pixel_values = pixel_values.to(self.device).to(torch.bfloat16)
            num_patches_list = [pixel_values.size(0)]
        else:
            pixel_values = None
            num_patches_list = []
            grid_ratio = None

        response, history = self.model.chat(
            self.tokenizer,
            pixel_values=pixel_values,
            num_patches_list=num_patches_list,
            question=question,
            generation_config=self.kwargs,
            history=history,
            return_history=True,
            grid_ratio=grid_ratio if 'grid_ratio' in locals() else None
        )

        response = re.sub(self.reverse_pattern, self.reverse_replacement, response)

        return response

    def chat_inner(self, message, dataset=None):
        self.set_max_num(dataset)

        if self.version in ['V1.1', 'V1.2']:
            raise ValueError(f'Unsupported version for Multi-Turn: {self.version}')
        elif self.version == 'V1.5':
            raise ValueError(f'Unsupported version for Multi-Turn: {self.version}')
        elif self.version == 'V2.0':
            kwargs_default = dict(do_sample=False, max_new_tokens=512, top_p=None, num_beams=1)
            self.kwargs = kwargs_default
            return self.chat_inner_v2(message, dataset)
        else:
            raise ValueError(f'Unsupported version for Multi-Turn: {self.version}')
