# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import warnings
from typing import List, Optional, Tuple, Union

import torch.utils.checkpoint
import transformers
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers import LlamaForCausalLM, Qwen2ForCausalLM, Qwen3ForCausalLM, Qwen3MoeForCausalLM

from .configuration_internvl_chat import InternVLChatConfig
from .conversation import get_conv_template
from .modeling_intern_vit import InternVisionModel, has_flash_attn

logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "InternVisionModel",
        "Qwen3MoeDecoderLayer",
    ]

    # support transformers 4.51.+
    _tp_plan = ''

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config._attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            architecture: str = config.llm_config.architectures[0]
            if architecture == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif architecture == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            elif architecture == 'Qwen3MoeForCausalLM':
                self.language_model = Qwen3MoeForCausalLM(config.llm_config)
            elif architecture == 'Qwen3ForCausalLM':
                self.language_model = Qwen3ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{architecture} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        #     print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = min(selected.sum(), vit_embeds.size(0))
            input_embeds[selected][:n_token] = input_embeds[selected][:n_token] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values, grid_ratio=None):
        """
        Extract visual features with optional tile pruning.
        
        Args:
            pixel_values: [num_patches, C, H, W] - All image patches including thumbnail
            grid_ratio: (rows, cols) tuple - Grid structure for tile organization
        
        Returns:
            vit_embeds: [num_kept_patches, num_tokens, embed_dim] - Visual embeddings
        """
        import math
        from ..tile_pruning_utils import compute_tile_importance_scores, select_tiles_to_keep
        from ..debug_utils import get_debug_logger
        logger = get_debug_logger()
        
        # Check if pruning is enabled
        enable_pruning = getattr(self.config, 'enable_tile_pruning', False)
        keep_ratio = getattr(self.config, 'tile_keep_ratio', 0.5)
        
        num_patches = pixel_values.shape[0]
        
        # Pruning disabled or only 1 patch (no tiles to prune)
        if not enable_pruning or num_patches <= 1:
            # Original logic: encode all patches
            return self._encode_all_patches(pixel_values)
        
        # === PRUNING ENABLED ===
        
        # Step 1: Identify Global Thumbnail (last patch by convention)
        thumbnail = pixel_values[-1:, :, :, :]  # [1, C, H, W]
        local_tiles = pixel_values[:-1, :, :, :]  # [num_tiles, C, H, W]
        num_tiles = local_tiles.shape[0]
        
        if logger and logger.should_log():
            logger.log_stage("Tile Pruning - Input", {
                "total_patches": num_patches,
                "num_local_tiles": num_tiles,
                "has_thumbnail": True,
                "grid_ratio": str(grid_ratio) if grid_ratio else "unknown",
                "keep_ratio": keep_ratio,
                "pruning_enabled": enable_pruning
            })
        
        # Step 2: Encode Global Thumbnail with attention output
        # IMPORTANT: Flash Attention does not return attention weights
        # Temporarily disable Flash Attention to get attention maps
        original_flash_attn_states = []
        for layer in self.vision_model.encoder.layers:
            original_flash_attn_states.append(layer.attn.use_flash_attn)
            layer.attn.use_flash_attn = False
        
        try:
            with torch.no_grad():
                thumbnail_outputs = self.vision_model(
                    pixel_values=thumbnail,
                    output_hidden_states=True if self.select_layer != -1 else False,
                    output_attentions=True,  # CRITICAL: Get attention maps
                    return_dict=True
                )
            
            # Step 3: Compute tile importance scores from attention
            last_attention = thumbnail_outputs.attentions[-1]  # [1, num_heads, N, N]
        finally:
            # Restore original Flash Attention states
            for layer, orig_state in zip(self.vision_model.encoder.layers, original_flash_attn_states):
                layer.attn.use_flash_attn = orig_state
        
        # Determine grid size
        if grid_ratio is not None:
            grid_size = grid_ratio  # (rows, cols)
        else:
            # Fallback: infer square-ish grid from tile count
            side = int(math.sqrt(num_tiles))
            grid_size = (side, (num_tiles + side - 1) // side)
        
        tile_scores = compute_tile_importance_scores(last_attention, grid_size)  # [1, num_tiles]
        keep_mask, kept_indices = select_tiles_to_keep(tile_scores, keep_ratio)
        
        if logger and logger.should_log():
            logger.log_stage("Tile Pruning - Scores", {
                "tile_scores": tile_scores[0].cpu().tolist(),
                "kept_indices": kept_indices[0].cpu().tolist(),
                "num_kept": kept_indices.shape[1],
                "num_dropped": num_tiles - kept_indices.shape[1]
            })
        
        # Print tile pruning info (token stats will be printed later in generate())
        if logger:
            print(f"Tile Pruning: Kept {kept_indices.shape[1]}/{num_tiles} tiles ({kept_indices.shape[1]/num_tiles*100:.1f}%)")
        
        # Step 4: Select and encode only kept tiles
        kept_tiles_list = []
        for idx in kept_indices[0]:
            kept_tiles_list.append(local_tiles[idx:idx+1])
        
        if len(kept_tiles_list) > 0:
            kept_tiles = torch.cat(kept_tiles_list, dim=0)  # [K, C, H, W]
            
            # Encode kept tiles
            if self.select_layer == -1:
                kept_embeds = self.vision_model(
                    pixel_values=kept_tiles,
                    output_hidden_states=False,
                    return_dict=True
                ).last_hidden_state
            else:
                kept_embeds = self.vision_model(
                    pixel_values=kept_tiles,
                    output_hidden_states=True,
                    return_dict=True
                ).hidden_states[self.select_layer]
        else:
            # Edge case: no tiles kept (shouldn't happen with max(1, ...))
            kept_embeds = torch.empty(0, 0, self.vision_model.config.hidden_size, device=pixel_values.device)
        
        # Step 5: Process thumbnail embedding
        if self.select_layer == -1:
            thumbnail_embed = thumbnail_outputs.last_hidden_state
        else:
            thumbnail_embed = thumbnail_outputs.hidden_states[self.select_layer]
        
        # Step 6: Concatenate kept tiles + thumbnail
        all_embeds = torch.cat([kept_embeds, thumbnail_embed], dim=0)  # [K+1, seq_len, embed_dim]
        
        # Apply standard post-processing (remove CLS, pixel shuffle, MLP)
        all_embeds = all_embeds[:, 1:, :]  # Remove CLS token
        
        h = w = int(all_embeds.shape[1] ** 0.5)
        all_embeds = all_embeds.reshape(all_embeds.shape[0], h, w, -1)
        all_embeds = self.pixel_shuffle(all_embeds, scale_factor=self.downsample_ratio)
        all_embeds = all_embeds.reshape(all_embeds.shape[0], -1, all_embeds.shape[-1])
        all_embeds = self.mlp1(all_embeds)
        
        if logger and logger.should_log():
            logger.log_stage("Tile Pruning - Output", {
                "output_shape": str(tuple(all_embeds.shape)),
                "tokens_per_patch": all_embeds.shape[1],
                "total_vision_tokens": all_embeds.shape[0] * all_embeds.shape[1]
            })
        
        return all_embeds
    
    def _encode_all_patches(self, pixel_values):
        """Original extract_feature logic without pruning."""
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True
            ).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True
            ).hidden_states[self.select_layer]
        
        vit_embeds = vit_embeds[:, 1:, :]
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        
        return vit_embeds

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep.strip())[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False, grid_ratio=None):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            grid_ratio=grid_ratio,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            grid_ratio: Optional[tuple] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        # [DEBUG] Get debug logger
        try:
            from ..debug_utils import get_debug_logger
            logger = get_debug_logger()
        except:
            logger = None
        
        # [DEBUG] Stage 3 - Input: Log generate input information
        if logger and logger.should_log():
            logger.log_stage("3. ViT Encoding - Input", {
                "pixel_values_shape": str(tuple(pixel_values.shape)) if pixel_values is not None else "None",
                "input_ids_shape": str(tuple(input_ids.shape)) if input_ids is not None else "None",
                "attention_mask_shape": str(tuple(attention_mask.shape)) if attention_mask is not None else "None",
                "batch_size": input_ids.shape[0] if input_ids is not None else 0,
                "sequence_length": input_ids.shape[1] if input_ids is not None else 0,
                "has_visual_features": visual_features is not None
            })

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values, grid_ratio=grid_ratio)
            
            # [DEBUG] Stage 3 - Output: Log ViT encoding output
            if logger and logger.should_log():
                logger.log_tensor("vit_embeds", vit_embeds)
                logger.log_stage("3. ViT Encoding - Output", {
                    "vit_embeds_shape": str(tuple(vit_embeds.shape)),
                    "vit_batch_size": pixel_values.shape[0],
                    "num_vit_tokens_per_patch": vit_embeds.shape[1],
                    "embedding_dim": vit_embeds.shape[2],
                    "total_vit_tokens": vit_embeds.shape[0] * vit_embeds.shape[1],
                    "memory_mb": f"{vit_embeds.element_size() * vit_embeds.nelement() / 1024 / 1024:.2f}"
                })
            
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            
            # [DEBUG] Stage 4 - Before: Log text embeddings before fusion
            if logger and logger.should_log():
                logger.log_tensor("text_embeds_before_fusion", input_embeds)
            
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids_flat = input_ids.reshape(B * N)
            selected = (input_ids_flat == self.img_context_token_id)
            num_img_tokens = selected.sum().item()
            
            # Calculate actual number of vision tokens after potential pruning
            actual_vit_tokens = vit_embeds.shape[0] * vit_embeds.shape[1]
            
            # Calculate real text tokens (excluding IMG_CONTEXT placeholders)
            real_text_tokens = B * N - num_img_tokens
            
            # Handle Tile Pruning: actual tokens may be less than placeholder tokens
            if actual_vit_tokens < num_img_tokens:
                # Tile Pruning is active - only replace the first N positions
                selected_indices = torch.where(selected)[0]
                # Create a new selection mask with only the first actual_vit_tokens positions
                new_selected = torch.zeros_like(selected)
                new_selected[selected_indices[:actual_vit_tokens]] = True
                selected = new_selected
                num_img_tokens_used = actual_vit_tokens
                # Unused placeholders remain as initial embeddings (padding-like)
                unused_placeholders = num_img_tokens - actual_vit_tokens
            else:
                num_img_tokens_used = num_img_tokens
                unused_placeholders = 0
            
            # [DEBUG] Stage 4 - Before: Log fusion information
            if logger and logger.should_log():
                logger.log_stage("4. Embedding Fusion - Before", {
                    "text_embeds_shape": f"({B}, {N}, {C})",
                    "vit_embeds_shape": str(tuple(vit_embeds.shape)),
                    "num_img_tokens_in_prompt": num_img_tokens,
                    "actual_vit_tokens": actual_vit_tokens,
                    "num_img_tokens_to_replace": num_img_tokens_used,
                    "unused_placeholders": unused_placeholders,
                    "real_text_tokens": real_text_tokens,
                    "total_tokens": B * N,
                    "pruning_active": actual_vit_tokens < num_img_tokens
                })
            
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
            
            # [DEBUG] Stage 4 - After: Log final embeddings after fusion
            if logger and logger.should_log():
                logger.log_tensor("final_embeds", input_embeds)
                logger.log_stage("4. Embedding Fusion - After", {
                    "final_embeds_shape": str(tuple(input_embeds.shape)),
                    "memory_mb": f"{input_embeds.element_size() * input_embeds.nelement() / 1024 / 1024:.2f}",
                    "breakdown": f"real_text({real_text_tokens}) + vision({num_img_tokens_used}) + padding({unused_placeholders})",
                    "real_text_tokens": real_text_tokens,
                    "vision_tokens": num_img_tokens_used,
                    "unused_placeholders": unused_placeholders,
                    "total_tokens": B * N
                })
            
            # [TOKEN STATS] Always print token statistics for every sample
            if logger:
                # Check if pruning was active
                pruning_active = actual_vit_tokens < num_img_tokens
                logger.print_token_stats(
                    vision_tokens=num_img_tokens_used,
                    text_tokens=real_text_tokens,
                    dataset=getattr(logger, 'current_dataset', None),
                    vision_tokens_before_pruning=num_img_tokens if pruning_active else None,
                    pruning_enabled=pruning_active
                )
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        # [DEBUG] Stage 5: Log LLM input
        if logger and logger.should_log():
            logger.log_stage("5. LLM Input", {
                "inputs_embeds_shape": str(tuple(input_embeds.shape)),
                "attention_mask_shape": str(tuple(attention_mask.shape)) if attention_mask is not None else "None",
                "attention_mask_sum": attention_mask.sum().item() if attention_mask is not None else "N/A",
                "total_tokens": input_embeds.shape[1],
                "batch_size": input_embeds.shape[0]
            })
            
            if attention_mask is not None:
                logger.log_attention_mask(attention_mask)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, value):
        return self.language_model.set_output_embeddings(value)
