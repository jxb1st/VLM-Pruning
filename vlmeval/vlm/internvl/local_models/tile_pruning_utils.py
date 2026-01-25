# --------------------------------------------------------
# InternVL Tile Pruning Utilities
# Attention-based tile pruning for reducing vision token redundancy
# --------------------------------------------------------

import torch
import torch.nn.functional as F
import math


def compute_tile_importance_scores(attention_map, grid_size):
    """
    Compute importance score for each tile based on Global Thumbnail attention.
    
    Args:
        attention_map: [B, Num_Heads, N_Tokens, N_Tokens] - Last layer attention from ViT
        grid_size: (rows, cols) tuple - e.g., (2, 3) for 2x3 tile grid
    
    Returns:
        tile_scores: [B, num_tiles] - Importance score for each tile
    """
    B, H, N, _ = attention_map.shape
    
    # Strategy: Use [CLS] token's attention to spatial tokens
    # CLS token is at index 0, spatial tokens are 1:N
    cls_attention = attention_map[:, :, 0, 1:]  # [B, H, N-1]
    
    # Average across attention heads
    cls_attention = cls_attention.mean(dim=1)  # [B, N-1]
    
    # Reshape to 2D spatial grid (e.g., 32x32 patches for 448x448 image)
    num_spatial_tokens = cls_attention.shape[1]
    side_length = int(math.sqrt(num_spatial_tokens))
    
    # Handle case where sqrt is not exact (shouldn't happen with standard ViT)
    if side_length * side_length != num_spatial_tokens:
        # Try to find closest square or rectangular arrangement
        side_length = int(math.ceil(math.sqrt(num_spatial_tokens)))
        # Pad if necessary
        pad_size = side_length * side_length - num_spatial_tokens
        if pad_size > 0:
            cls_attention = F.pad(cls_attention, (0, pad_size), value=0)
    
    spatial_attention = cls_attention.view(B, side_length, side_length)  # [B, H, W]
    
    # Downsample to tile grid resolution
    # E.g., from 32x32 → 2x3 (average pooling within each tile region)
    tile_scores = F.adaptive_avg_pool2d(
        spatial_attention.unsqueeze(1),  # [B, 1, H, W]
        grid_size  # (rows, cols)
    )  # [B, 1, rows, cols]
    
    # Flatten to [B, num_tiles]
    num_tiles = grid_size[0] * grid_size[1]
    tile_scores = tile_scores.view(B, num_tiles)
    
    return tile_scores


def select_tiles_to_keep(tile_scores, keep_ratio=0.5):
    """
    Select top-K tiles based on importance scores.
    
    Args:
        tile_scores: [B, num_tiles] - Importance scores
        keep_ratio: float - Fraction of tiles to keep (0.0-1.0)
    
    Returns:
        keep_mask: [B, num_tiles] - Boolean mask, True for tiles to keep
        kept_indices: [B, K] - Indices of kept tiles (sorted by score descending)
    """
    B, num_tiles = tile_scores.shape
    num_keep = max(1, int(num_tiles * keep_ratio))  # Keep at least 1 tile
    
    # Get indices of top-K tiles (sorted by score, highest first)
    _, kept_indices = torch.topk(tile_scores, num_keep, dim=1, sorted=True)  # [B, K]
    
    # Create boolean mask
    keep_mask = torch.zeros_like(tile_scores, dtype=torch.bool)  # [B, num_tiles]
    keep_mask.scatter_(1, kept_indices, True)
    
    return keep_mask, kept_indices

