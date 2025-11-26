# Drywall Prompted Segmentation

Text-conditioned segmentation for automated drywall quality inspection using CLIPSeg.

## Overview

Fine-tuned CLIPSeg model to segment drywall defects based on natural language prompts:
- **"segment taping area"** / **"segment joint"** / **"segment drywall seam"** - detects drywall joints/seams
- **"segment crack"** / **"segment wall crack"** - detects wall cracks

## Results

| Prompt | Count | mIoU | mDice |
|--------|-------|------|-------|
| segment taping area | 250 | 0.5837 | 0.7243 |
| segment joint | 250 | 0.5857 | 0.7262 |
| segment drywall seam | 250 | 0.5798 | 0.7212 |
| segment crack | 201 | 0.4873 | 0.6329 |
| segment wall crack | 201 | 0.4825 | 0.6284 |
| **Overall Average** | | **0.5438** | **0.6866** |

**Task-level Performance:**
- Taping Area: mIoU 0.5831, mDice 0.7239
- Cracks: mIoU 0.4849, mDice 0.6307

## Project Structure
```
├── 10x_assignment.ipynb          # Training notebook
├── 10x_assignment_Report.pdf     # Detailed report
├── predictions/                  # Validation predictions (PNG format)
├── images/                       # Sample visualizations
└── README.md
```

## Quick Start
```python
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torch
from PIL import Image

# Load model
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
model.load_state_dict(torch.load('model_checkpoints/best.pt'))
model.eval()

# Run inference
image = Image.open('path/to/image.jpg')
inputs = processor(text=["segment crack"], images=[image], return_tensors="pt")
outputs = model(**inputs)
mask = torch.sigmoid(outputs.logits).squeeze().numpy()
```

## Training Details

- **Model**: CLIPSeg (CIDAS/clipseg-rd64-refined, ~90M parameters)
- **Datasets**: 
  - Taping areas: 250 validation samples
  - Cracks: 201 validation samples
- **Training time**: ~25 minutes on Colab T4 GPU
- **Inference**: ~50ms per image
- **Hyperparameters**: Batch size 64, LR 1e-4, 15 epochs
- **Seed**: 42

## Key Findings

- Taping/joint detection achieves 0.72 Dice due to clear boundaries and larger features
- Crack detection is more challenging at 0.63 Dice due to thin structures
- Main limitation: 352x352 internal resolution causes fragmentation on hairline cracks
- Model generalizes well across prompt variations with consistent performance

## Requirements
```bash
pip install transformers torch torchvision pillow numpy opencv-python
```

## Output Format

Predictions saved as single-channel PNG files (values: 0 or 255) with naming format:
```
<image_id>__<prompt_with_underscores>.png
```

---

**Full report available in**: `10x_assignment_Report.pdf`
