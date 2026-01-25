#!/usr/bin/env python
"""
Tile Pruning Visualization Script
Visualizes the tile pruning process and intermediate results.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path
import math

from vlmeval.vlm.internvl import InternVLChat
from vlmeval.vlm.internvl.utils import dynamic_preprocess, load_image
from vlmeval.vlm.internvl.local_models.tile_pruning_utils import compute_tile_importance_scores, select_tiles_to_keep

# Create output directory
output_dir = Path("tile_pruning_visualizations")
output_dir.mkdir(exist_ok=True)

print("="*80)
print("Tile Pruning Visualization Script")
print("="*80)

# Configuration
test_image_path = '/gpfs/projects/embodied3d/jianxu/vlm_pruning/VLMEvalKit/assets/demo.jpeg'
keep_ratio = 0.5

print(f"\nConfiguration:")
print(f"  Test image: {test_image_path}")
print(f"  Keep ratio: {keep_ratio}")
print(f"  Output directory: {output_dir}")


def extract_attention_and_scores(model, pixel_values, grid_ratio, keep_ratio):
    """
    Extract attention maps and compute tile scores.
    
    Returns:
        all_attentions: List of attention tensors from all layers
        tile_scores: Tensor of tile importance scores
        kept_indices: Tensor of indices of kept tiles
        last_attention: Last layer attention map
        cls_attention_2d: 2D spatial attention map
    """
    print("\n" + "="*80)
    print("Extracting Attention Maps and Computing Tile Scores")
    print("="*80)
    
    with torch.no_grad():
        # Separate thumbnail and tiles
        thumbnail = pixel_values[-1:, :, :, :]
        local_tiles = pixel_values[:-1, :, :, :]
        num_tiles = local_tiles.shape[0]
        
        print(f"Thumbnail shape: {thumbnail.shape}")
        print(f"Local tiles shape: {local_tiles.shape}")
        print(f"Number of local tiles: {num_tiles}")
        
        # Temporarily disable Flash Attention to get attention maps
        vision_model = model.model.vision_model
        original_flash_attn_states = []
        for layer in vision_model.encoder.layers:
            original_flash_attn_states.append(layer.attn.use_flash_attn)
            layer.attn.use_flash_attn = False
        
        try:
            # Encode thumbnail with attention
            thumbnail_outputs = vision_model(
                pixel_values=thumbnail,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True
            )
            
            # Get attention maps from all layers
            all_attentions = thumbnail_outputs.attentions
            last_attention = all_attentions[-1]  # [1, num_heads, N, N]
            
            print(f"\nAttention maps extracted:")
            print(f"  Number of layers: {len(all_attentions)}")
            print(f"  Last layer attention shape: {last_attention.shape}")
            
            # Extract CLS attention to spatial tokens and reshape to 2D
            B, H, N, _ = last_attention.shape
            cls_attention = last_attention[0, :, 0, 1:]  # [num_heads, num_spatial_tokens]
            cls_attention_avg = cls_attention.mean(dim=0).float().cpu().numpy()  # [num_spatial_tokens]
            
            # Reshape to 2D spatial grid
            num_spatial = cls_attention_avg.shape[0]
            side_length = int(math.sqrt(num_spatial))
            cls_attention_2d = cls_attention_avg.reshape(side_length, side_length)
            
            print(f"  Spatial attention shape: {cls_attention_2d.shape}")
            
            # Compute tile scores
            tile_scores = compute_tile_importance_scores(last_attention, grid_ratio)
            tile_scores_np = tile_scores[0].cpu().to(torch.float32).numpy()
            
            print(f"\nTile Importance Scores:")
            for i, score in enumerate(tile_scores_np):
                print(f"  Tile {i}: {score:.6f}")
            
            # Select tiles to keep
            keep_mask, kept_indices = select_tiles_to_keep(tile_scores, keep_ratio)
            kept_indices_np = kept_indices[0].cpu().numpy()
            
            print(f"\nKept tiles: {sorted(kept_indices_np.tolist())}")
            print(f"Dropped tiles: {sorted([i for i in range(num_tiles) if i not in kept_indices_np])}")
            
        finally:
            # Restore Flash Attention
            for layer, orig_state in zip(vision_model.encoder.layers, original_flash_attn_states):
                layer.attn.use_flash_attn = orig_state
    
    return all_attentions, tile_scores, kept_indices, last_attention, cls_attention_2d


def visualize_tile_division(original_image, processed_images, grid_ratio, output_path):
    """Create visualization showing tile division."""
    print("\n" + "="*80)
    print("Visualizing Tile Division")
    print("="*80)
    
    rows, cols = grid_ratio
    num_tiles = rows * cols
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Original image with grid overlay
    ax = axes[0]
    ax.imshow(original_image)
    ax.set_title(f'Original Image with {rows}×{cols} Tile Grid', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Draw grid
    img_width, img_height = original_image.size
    tile_width = img_width / cols
    tile_height = img_height / rows
    
    for i in range(rows + 1):
        y = i * tile_height
        ax.plot([0, img_width], [y, y], 'r-', linewidth=2)
    for j in range(cols + 1):
        x = j * tile_width
        ax.plot([x, x], [0, img_height], 'r-', linewidth=2)
    
    # Add tile numbers
    for i in range(rows):
        for j in range(cols):
            tile_idx = i * cols + j
            cx = (j + 0.5) * tile_width
            cy = (i + 0.5) * tile_height
            ax.text(cx, cy, f'Tile {tile_idx}', 
                   ha='center', va='center',
                   fontsize=20, fontweight='bold',
                   color='yellow', 
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Right: Show all tiles
    ax = axes[1]
    ax.axis('off')
    ax.set_title(f'All {len(processed_images)} Patches (Local Tiles + Thumbnail)', 
                fontsize=14, fontweight='bold')
    
    # Arrange tiles in grid
    n_cols_display = 4
    n_rows_display = (len(processed_images) + n_cols_display - 1) // n_cols_display
    tile_mosaic = np.zeros((n_rows_display * 448, n_cols_display * 448, 3), dtype=np.uint8)
    
    for idx, tile_img in enumerate(processed_images):
        row_idx = idx // n_cols_display
        col_idx = idx % n_cols_display
        tile_array = np.array(tile_img)
        tile_mosaic[row_idx*448:(row_idx+1)*448, col_idx*448:(col_idx+1)*448] = tile_array
        
    ax.imshow(tile_mosaic)
    
    # Add labels
    for idx in range(len(processed_images)):
        row_idx = idx // n_cols_display
        col_idx = idx % n_cols_display
        cx = (col_idx + 0.5) * 448
        cy = (row_idx + 0.1) * 448
        label = f'Tile {idx}' if idx < num_tiles else 'Thumbnail'
        ax.text(cx, cy, label,
               ha='center', va='top',
               fontsize=16, fontweight='bold',
               color='yellow',
               bbox=dict(boxstyle='round', facecolor='blue' if idx < num_tiles else 'green', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def visualize_attention_maps(all_attentions, output_path):
    """Show attention evolution across layers."""
    print("\n" + "="*80)
    print("Visualizing Attention Maps")
    print("="*80)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Attention Maps from Different Layers', fontsize=16, fontweight='bold')
    
    # Show attention from last 8 layers
    num_layers_to_show = min(8, len(all_attentions))
    start_layer = len(all_attentions) - num_layers_to_show
    
    for idx, layer_idx in enumerate(range(start_layer, len(all_attentions))):
        ax = axes[idx // 4, idx % 4]
        
        # Extract CLS attention
        attn = all_attentions[layer_idx][0, :, 0, 1:].mean(dim=0).float().cpu().numpy()
        num_spatial = attn.shape[0]
        side_length = int(math.sqrt(num_spatial))
        attn_2d = attn.reshape(side_length, side_length)
        
        im = ax.imshow(attn_2d, cmap='hot', interpolation='bilinear')
        ax.set_title(f'Layer {layer_idx}', fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def visualize_tile_scores(cls_attention_2d, tile_scores, kept_indices, grid_ratio, keep_ratio, output_path):
    """Show attention->scores pipeline."""
    print("\n" + "="*80)
    print("Visualizing Tile Importance Scores")
    print("="*80)
    
    rows, cols = grid_ratio
    num_tiles = rows * cols
    tile_scores_np = tile_scores[0].cpu().to(torch.float32).numpy()
    kept_indices_np = kept_indices[0].cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # (a) Spatial attention heatmap
    ax = axes[0]
    side_length = cls_attention_2d.shape[0]
    im = ax.imshow(cls_attention_2d, cmap='hot', interpolation='bilinear')
    ax.set_title(f'CLS Token Attention Map\n({side_length}×{side_length} spatial tokens)',
                fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # (b) Tile scores heatmap
    ax = axes[1]
    tile_scores_2d = tile_scores_np.reshape(rows, cols)
    im = ax.imshow(tile_scores_2d, cmap='hot', interpolation='nearest')
    ax.set_title(f'Tile Importance Scores\n({rows}×{cols} tile grid)',
                fontsize=12, fontweight='bold')
    
    # Add values on heatmap
    for i in range(rows):
        for j in range(cols):
            tile_idx = i * cols + j
            text = ax.text(j, i, f'{tile_scores_np[tile_idx]:.4f}',
                          ha="center", va="center", color="white", fontweight='bold')
    
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticklabels([f'Col {j}' for j in range(cols)])
    ax.set_yticklabels([f'Row {i}' for i in range(rows)])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # (c) Bar chart of tile scores
    ax = axes[2]
    colors = ['green' if i in kept_indices_np else 'red' for i in range(num_tiles)]
    bars = ax.bar(range(num_tiles), tile_scores_np, color=colors)
    ax.set_xlabel('Tile Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Tile Scores (keep_ratio={keep_ratio})\nGreen=Keep, Red=Drop',
                fontsize=12, fontweight='bold')
    ax.set_xticks(range(num_tiles))
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def visualize_selection(original_image, tile_scores, kept_indices, grid_ratio, processed_images, output_path):
    """Show kept vs dropped tiles."""
    print("\n" + "="*80)
    print("Visualizing Kept vs Dropped Tiles")
    print("="*80)
    
    rows, cols = grid_ratio
    num_tiles = rows * cols
    tile_scores_np = tile_scores[0].cpu().to(torch.float32).numpy()
    kept_indices_np = kept_indices[0].cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # (a) Original image with kept/dropped tiles highlighted
    ax = axes[0]
    ax.imshow(original_image)
    ax.set_title(f'Kept (Green) vs Dropped (Red) Tiles', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Draw tiles with colors
    img_width, img_height = original_image.size
    tile_width = img_width / cols
    tile_height = img_height / rows
    
    for i in range(rows):
        for j in range(cols):
            tile_idx = i * cols + j
            x = j * tile_width
            y = i * tile_height
            
            # Determine color
            if tile_idx in kept_indices_np:
                color = 'green'
                alpha = 0.3
                label = 'KEEP'
            else:
                color = 'red'
                alpha = 0.5
                label = 'DROP'
            
            # Draw rectangle
            rect = patches.Rectangle((x, y), tile_width, tile_height,
                                    linewidth=3, edgecolor=color, facecolor=color, alpha=alpha)
            ax.add_patch(rect)
            
            # Add label
            cx = x + tile_width / 2
            cy = y + tile_height / 2
            ax.text(cx, cy, f'{label}\nTile {tile_idx}\n{tile_scores_np[tile_idx]:.4f}',
                   ha='center', va='center',
                   fontsize=14, fontweight='bold',
                   color='white',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    
    # (b) Show kept tiles only
    ax = axes[1]
    ax.axis('off')
    ax.set_title(f'Kept Tiles Only ({len(kept_indices_np)}/{num_tiles})', 
                fontsize=14, fontweight='bold')
    
    # Arrange kept tiles
    n_kept = len(kept_indices_np)
    n_cols_kept = min(3, n_kept)
    n_rows_kept = (n_kept + n_cols_kept - 1) // n_cols_kept
    kept_mosaic = np.zeros((n_rows_kept * 448, n_cols_kept * 448, 3), dtype=np.uint8)
    
    for idx, tile_idx in enumerate(sorted(kept_indices_np)):
        row_idx = idx // n_cols_kept
        col_idx = idx % n_cols_kept
        tile_img = processed_images[tile_idx]
        tile_array = np.array(tile_img)
        kept_mosaic[row_idx*448:(row_idx+1)*448, col_idx*448:(col_idx+1)*448] = tile_array
        
        # Add label
        cx = (col_idx + 0.5) * 448
        cy = (row_idx + 0.1) * 448
        ax.text(cx, cy, f'Tile {tile_idx}\nScore: {tile_scores_np[tile_idx]:.4f}',
               ha='center', va='top',
               fontsize=14, fontweight='bold',
               color='yellow',
               bbox=dict(boxstyle='round', facecolor='green', alpha=0.8))
    
    ax.imshow(kept_mosaic)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_summary(original_image, processed_images, cls_attention_2d, tile_scores, 
                   kept_indices, grid_ratio, keep_ratio, output_path):
    """Create comprehensive summary visualization."""
    print("\n" + "="*80)
    print("Creating Summary Visualization")
    print("="*80)
    
    rows, cols = grid_ratio
    num_tiles = rows * cols
    tile_scores_np = tile_scores[0].cpu().to(torch.float32).numpy()
    kept_indices_np = kept_indices[0].cpu().numpy()
    tile_scores_2d = tile_scores_np.reshape(rows, cols)
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # (1) Original image
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(original_image)
    ax.set_title('1. Original Image', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # (2) Tile grid
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(original_image)
    img_width, img_height = original_image.size
    tile_width = img_width / cols
    tile_height = img_height / rows
    for i in range(rows + 1):
        y = i * tile_height
        ax.plot([0, img_width], [y, y], 'r-', linewidth=2)
    for j in range(cols + 1):
        x = j * tile_width
        ax.plot([x, x], [0, img_height], 'r-', linewidth=2)
    ax.set_title(f'2. Tile Division ({rows}×{cols})', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # (3) Thumbnail
    ax = fig.add_subplot(gs[0, 2])
    thumbnail_img = processed_images[-1]
    ax.imshow(thumbnail_img)
    ax.set_title('3. Global Thumbnail', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # (4) Attention map
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(cls_attention_2d, cmap='hot', interpolation='bilinear')
    ax.set_title('4. CLS Attention Map', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # (5) Tile scores
    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(tile_scores_2d, cmap='hot', interpolation='nearest')
    ax.set_title('5. Tile Importance Scores', fontsize=12, fontweight='bold')
    for i in range(rows):
        for j in range(cols):
            tile_idx = i * cols + j
            ax.text(j, i, f'{tile_idx}\n{tile_scores_np[tile_idx]:.3f}',
                   ha="center", va="center", color="white", fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # (6) Score bars
    ax = fig.add_subplot(gs[1, 2])
    colors = ['green' if i in kept_indices_np else 'red' for i in range(num_tiles)]
    ax.bar(range(num_tiles), tile_scores_np, color=colors)
    ax.set_title(f'6. Keep (Green) vs Drop (Red)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Tile Index')
    ax.set_ylabel('Score')
    ax.grid(axis='y', alpha=0.3)
    
    # (7) Kept tiles overlay
    ax = fig.add_subplot(gs[2, :2])
    ax.imshow(original_image)
    for i in range(rows):
        for j in range(cols):
            tile_idx = i * cols + j
            x = j * tile_width
            y = i * tile_height
            if tile_idx in kept_indices_np:
                color, alpha, label = 'green', 0.3, 'KEEP'
            else:
                color, alpha, label = 'red', 0.5, 'DROP'
            rect = patches.Rectangle((x, y), tile_width, tile_height,
                                    linewidth=3, edgecolor=color, facecolor=color, alpha=alpha)
            ax.add_patch(rect)
            cx, cy = x + tile_width / 2, y + tile_height / 2
            ax.text(cx, cy, f'{label}\n{tile_idx}',
                   ha='center', va='center', fontsize=16, fontweight='bold',
                   color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    ax.set_title('7. Final Selection Overlay', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # (8) Statistics
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    stats_text = f"""
TILE PRUNING SUMMARY
{'='*30}

Grid: {rows} × {cols} = {num_tiles} tiles
Keep Ratio: {keep_ratio:.1%}
Kept: {len(kept_indices_np)} tiles
Dropped: {num_tiles - len(kept_indices_np)} tiles

Vision Tokens:
  Original: {num_tiles * 256 + 256:,}
  Pruned: {len(kept_indices_np) * 256 + 256:,}
  Reduction: {(1 - (len(kept_indices_np) + 1) / (num_tiles + 1)):.1%}

Kept Tile IDs:
  {sorted(kept_indices_np.tolist())}

Dropped Tile IDs:
  {sorted([i for i in range(num_tiles) if i not in kept_indices_np])}
"""
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Tile Pruning Pipeline Visualization', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


# Main execution
if __name__ == '__main__':
    # Load test image
    print(f"\nLoading test image: {test_image_path}")
    original_image = Image.open(test_image_path).convert('RGB')
    print(f"Original image size: {original_image.size}")
    
    # Load model
    print("\nLoading InternVL3.5-30B-A3B...")
    model = InternVLChat(model_path='OpenGVLab/InternVL3_5-30B-A3B', version='V2.0')
    
    # Enable tile pruning
    model.model.config.enable_tile_pruning = True
    model.model.config.tile_keep_ratio = keep_ratio
    print(f"Tile pruning enabled: keep_ratio={keep_ratio}")
    
    # Preprocess image to get tiles and grid_ratio
    print("\nPreprocessing image...")
    processed_images, target_ratio = dynamic_preprocess(
        original_image, 
        image_size=448, 
        use_thumbnail=True, 
        max_num=6
    )
    rows, cols = target_ratio
    num_tiles = rows * cols
    print(f"Grid size: {rows} rows × {cols} cols = {num_tiles} tiles")
    print(f"Total patches (including thumbnail): {len(processed_images)}")
    
    # Load image through the pipeline
    pixel_values, grid_ratio = load_image(test_image_path, max_num=6, upscale=False)
    pixel_values = pixel_values.to(model.model.device).to(torch.bfloat16)
    
    # Extract attention maps and tile scores
    all_attentions, tile_scores, kept_indices, last_attention, cls_attention_2d = \
        extract_attention_and_scores(model, pixel_values, grid_ratio, keep_ratio)
    
    # Generate all visualizations
    print("\n" + "="*80)
    print("Generating Visualizations")
    print("="*80)
    
    # Visualization 1: Tile Division
    visualize_tile_division(
        original_image, processed_images, grid_ratio,
        output_dir / '01_tile_division.png'
    )
    
    # Visualization 2: Attention Maps
    visualize_attention_maps(
        all_attentions,
        output_dir / '02_attention_maps.png'
    )
    
    # Visualization 3: Tile Scores
    visualize_tile_scores(
        cls_attention_2d, tile_scores, kept_indices, grid_ratio, keep_ratio,
        output_dir / '03_tile_scores.png'
    )
    
    # Visualization 4: Kept vs Dropped
    visualize_selection(
        original_image, tile_scores, kept_indices, grid_ratio, processed_images,
        output_dir / '04_kept_vs_dropped.png'
    )
    
    # Visualization 5: Summary
    create_summary(
        original_image, processed_images, cls_attention_2d, tile_scores, 
        kept_indices, grid_ratio, keep_ratio,
        output_dir / '05_summary.png'
    )
    
    # Final summary
    print("\n" + "="*80)
    print("Visualization Complete!")
    print("="*80)
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  1. 01_tile_division.png - Tile grid and all patches")
    print(f"  2. 02_attention_maps.png - Attention from multiple layers")
    print(f"  3. 03_tile_scores.png - Tile importance scores")
    print(f"  4. 04_kept_vs_dropped.png - Which tiles are kept/dropped")
    print(f"  5. 05_summary.png - Complete pipeline summary")
    print("\n" + "="*80)

