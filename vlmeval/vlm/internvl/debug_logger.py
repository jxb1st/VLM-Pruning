"""
InternVL Debug Logger for Token Pruning Research

This module provides detailed logging of tensor shapes, memory usage, and 
data flow through different stages of InternVL inference pipeline.
"""

import os
import torch
from datetime import datetime
from pathlib import Path


class InternVLDebugLogger:
    """Logger for debugging InternVL token flow and pruning research."""
    
    def __init__(self, log_dir="./debug_logs"):
        """Initialize the debug logger.
        
        Args:
            log_dir: Directory to save log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"internvl_debug_{timestamp}.log"
        
        # Sample counter - only log first sample
        self.sample_count = 0
        self.max_samples_to_log = 1
        
        # Global sample counter for token stats (always increment)
        self.total_sample_count = 0
        
        # Write header
        self._write_header()
    
    def _write_header(self):
        """Write header to log file."""
        header = f"""
{'='*80}
InternVL Token Pruning Debug Log
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{'='*80}

"""
        with open(self.log_file, 'w') as f:
            f.write(header)
    
    def should_log(self):
        """Check if current sample should be logged.
        
        Returns:
            bool: True if should log, False otherwise
        """
        return self.sample_count < self.max_samples_to_log
    
    def increment_sample(self):
        """Increment sample counter."""
        self.sample_count += 1
    
    def log_stage(self, stage_name, data_dict):
        """Log a processing stage with its data.
        
        Args:
            stage_name: Name of the processing stage
            data_dict: Dictionary containing stage data
        """
        if not self.should_log():
            return
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Format message
        message = f"\n{'='*80}\n"
        message += f"STAGE: {stage_name}\n"
        message += f"Time: {timestamp}\n"
        message += f"{'-'*80}\n"
        
        for key, value in data_dict.items():
            message += f"{key}: {value}\n"
        
        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(message)
        
        # Print to console
        print(message, end='')
    
    def log_tensor(self, name, tensor):
        """Log detailed tensor information.
        
        Args:
            name: Name of the tensor
            tensor: PyTorch tensor to log
        """
        if not self.should_log():
            return
        
        if tensor is None:
            self.log_stage(f"TENSOR: {name}", {"value": "None"})
            return
        
        # Calculate memory
        memory_mb = tensor.element_size() * tensor.nelement() / (1024 * 1024)
        
        # Calculate statistics
        try:
            tensor_flat = tensor.flatten().float()
            stats = {
                "min": f"{tensor_flat.min().item():.4f}",
                "max": f"{tensor_flat.max().item():.4f}",
                "mean": f"{tensor_flat.mean().item():.4f}",
                "std": f"{tensor_flat.std().item():.4f}"
            }
        except:
            stats = {"note": "Statistics unavailable"}
        
        data = {
            "shape": str(tuple(tensor.shape)),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "memory_mb": f"{memory_mb:.2f} MB",
            "requires_grad": tensor.requires_grad,
            "is_contiguous": tensor.is_contiguous(),
            **stats
        }
        
        self.log_stage(f"TENSOR: {name}", data)
    
    def log_attention_mask(self, attention_mask):
        """Log attention mask details.
        
        Args:
            attention_mask: Attention mask tensor
        """
        if not self.should_log() or attention_mask is None:
            return
        
        valid_tokens = attention_mask.sum().item()
        total_tokens = attention_mask.numel()
        
        data = {
            "shape": str(tuple(attention_mask.shape)),
            "valid_tokens": valid_tokens,
            "total_tokens": total_tokens,
            "padding_tokens": total_tokens - valid_tokens,
            "padding_ratio": f"{(1 - valid_tokens/total_tokens)*100:.2f}%"
        }
        
        self.log_stage("ATTENTION MASK", data)
    
    def log_token_ids(self, name, token_ids, tokenizer=None):
        """Log token IDs with optional decoding.
        
        Args:
            name: Name/description of the tokens
            token_ids: Token ID tensor
            tokenizer: Optional tokenizer for decoding
        """
        if not self.should_log():
            return
        
        data = {
            "shape": str(tuple(token_ids.shape)),
            "num_tokens": token_ids.numel(),
            "unique_tokens": len(torch.unique(token_ids)),
        }
        
        if tokenizer is not None:
            try:
                # Decode first 50 tokens for preview
                preview_ids = token_ids.flatten()[:50].tolist()
                preview_text = tokenizer.decode(preview_ids, skip_special_tokens=False)
                data["preview"] = preview_text[:200] + "..." if len(preview_text) > 200 else preview_text
            except:
                data["preview"] = "Decode failed"
        
        self.log_stage(f"TOKEN IDS: {name}", data)
    
    def log_summary(self):
        """Log summary at the end."""
        if self.sample_count > 0:
            summary = f"""
{'='*80}
SUMMARY
{'='*80}
Total samples logged: {self.sample_count}
Log file: {self.log_file}
{'='*80}
"""
            with open(self.log_file, 'a') as f:
                f.write(summary)
            print(summary)
    
    def print_token_stats(self, vision_tokens, text_tokens, dataset=None, 
                          vision_tokens_before_pruning=None, pruning_enabled=False):
        """Print concise token statistics to terminal (always, not affected by should_log).
        
        Args:
            vision_tokens: Number of vision tokens (after pruning if enabled)
            text_tokens: Number of text tokens
            dataset: Optional dataset name
            vision_tokens_before_pruning: Number of vision tokens before pruning
            pruning_enabled: Whether tile pruning was active
        """
        self.total_sample_count += 1
        total = vision_tokens + text_tokens
        
        dataset_str = f" [{dataset}]" if dataset else ""
        
        # Base stats
        base_msg = f"Sample #{self.total_sample_count}{dataset_str} | Vision: {vision_tokens} | Text: {text_tokens} | Total: {total}"
        
        # Add pruning info if applicable
        if pruning_enabled and vision_tokens_before_pruning is not None and vision_tokens_before_pruning > vision_tokens:
            pruned_tokens = vision_tokens_before_pruning - vision_tokens
            reduction_pct = (pruned_tokens / vision_tokens_before_pruning) * 100
            pruning_msg = f" | Pruned: {pruned_tokens} tokens ({reduction_pct:.1f}% reduction, {vision_tokens_before_pruning}→{vision_tokens})"
            print(base_msg + pruning_msg)
        else:
            print(base_msg)


