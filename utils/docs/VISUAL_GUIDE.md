# Image Resizer - How It Works (Visual Guide)

## The Algorithm

The resizer uses a 3-step process to ensure perfect dimension matching while preserving image quality and aspect ratio.

---

## Step 1: Aspect Ratio Comparison

```
TARGET: 800 x 600          METHOD: 1600 x 900
Ratio: 800/600 = 1.33      Ratio: 1600/900 = 1.78

METHOD IS WIDER (1.78 > 1.33)
→ Scale to match WIDTH, then adjust HEIGHT
```

---

## Step 2: Scale to Fit (Preserve Aspect Ratio)

### Case A: Method is Wider (horizontal image)

```
Original Method: 1600 x 900
Target: 800 x 600

Scale to match width:
  scale_factor = 800 / 1600 = 0.5
  new_width = 800
  new_height = 900 * 0.5 = 450

Result: 800 x 450 (maintains 16:9 ratio)
```

**Visual:**
```
┌──────────────────────────┐
│                          │  1600 x 900
│      WIDE IMAGE          │
└──────────────────────────┘
              ↓ scale by 0.5
       ┌──────────────┐
       │ WIDE IMAGE   │  800 x 450
       └──────────────┘
```

### Case B: Method is Taller (vertical image)

```
Original Method: 600 x 1200
Target: 800 x 600

Scale to match height:
  scale_factor = 600 / 1200 = 0.5
  new_width = 600 * 0.5 = 300
  new_height = 600

Result: 300 x 600 (maintains 1:2 ratio)
```

**Visual:**
```
    ┌────────┐
    │        │
    │ TALL   │  600 x 1200
    │ IMAGE  │
    │        │
    └────────┘
         ↓ scale by 0.5
      ┌───┐
      │   │
      │ T │  300 x 600
      │ A │
      │ L │
      │ L │
      └───┘
```

---

## Step 3: Adjust to Exact Target Size

### Option A: PADDING (when scaled image is smaller)

**Example: 800x450 → 800x600**

```
Scaled image: 800 x 450
Target:       800 x 600
Difference:   0 x 150 (need 150px vertical padding)

Add 75px to top and 75px to bottom:

       ┌──────────────┐
       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ← 75px padding (black/white/color)
       ┌──────────────┐
       │              │
       │ WIDE IMAGE   │  800 x 450 (actual image)
       │              │
       └──────────────┘
       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ← 75px padding
       └──────────────┘
           800 x 600 ✓
```

**With different padding colors:**

```
Black (default):        White:              Custom (red):
┌──────────────┐       ┌──────────────┐     ┌──────────────┐
█████████████████       ░░░░░░░░░░░░░░░     ▓▓▓▓▓▓▓▓▓▓▓▓▓▓  (255,0,0)
│ image here   │       │ image here   │     │ image here   │
█████████████████       ░░░░░░░░░░░░░░░     ▓▓▓▓▓▓▓▓▓▓▓▓▓▓
└──────────────┘       └──────────────┘     └──────────────┘
```

### Option B: CROPPING (when scaled image is larger)

**Example: 1066x600 → 800x600**

```
Scaled image: 1066 x 600
Target:        800 x 600
Difference:    266 x 0 (need to crop 266px horizontally)

Crop 133px from left and 133px from right (center mode):

┌─────────────────────────────┐
│ :   CENTERED IMAGE    :     │  1066 x 600
│ :                     :     │
└─────────────────────────────┘
  ↑                       ↑
  133px                   133px
  crop                    crop

       ┌──────────────┐
       │ CENTER IMAGE │  800 x 600 ✓
       └──────────────┘
```

**Different crop modes:**

```
CENTER (default):              TOP:                    BOTTOM:
┌─────────────────┐           ┌─────────────────┐     ┌─────────────────┐
│ : keep this :   │           │ keep this area  │     │                 │
└─────────────────┘           └─────────────────┘     │                 │
  ↑           ↑               ↑                        └─────────────────┘
 crop       crop             crop bottom               keep this area ↑
 left       right                                      crop top

LEFT:                         RIGHT:
┌─────────────────┐           ┌─────────────────┐
│ keep this   :   │           │   : keep this   │
└─────────────────┘           └─────────────────┘
                ↑             ↑
              crop           crop
              right          left
```

---

## Complete Examples

### Example 1: Wide Panorama to Standard

```
INPUT:
Method: 2000 x 500 (panorama, 4:1 ratio)
Target:  800 x 600 (standard, 4:3 ratio)

STEP 1: Compare aspect ratios
Method ratio: 2000/500 = 4.0
Target ratio: 800/600 = 1.33
→ Method is WIDER

STEP 2: Scale to match width
Scale factor: 800/2000 = 0.4
New size: 800 x 200

STEP 3: Pad vertically
Need: 600 - 200 = 400px padding
Add: 200px top + 200px bottom

OUTPUT:
       ┌──────────────┐
       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ← 200px padding
       ┌──────────────┐
       │   PANORAMA   │ ← 800 x 200
       └──────────────┘
       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ← 200px padding
       └──────────────┘
           800 x 600 ✓
```

### Example 2: Portrait to Landscape

```
INPUT:
Method:  600 x 1200 (portrait, 1:2 ratio)
Target: 1920 x 1080 (landscape, 16:9 ratio)

STEP 1: Compare aspect ratios
Method ratio: 600/1200 = 0.5
Target ratio: 1920/1080 = 1.78
→ Method is TALLER

STEP 2: Scale to match height
Scale factor: 1080/1200 = 0.9
New size: 540 x 1080

STEP 3: Pad horizontally
Need: 1920 - 540 = 1380px padding
Add: 690px left + 690px right

OUTPUT:
    ┌────────────────────────────────┐
    │     │                    │     │
    │ 690 │    540 x 1080     │ 690 │
    │ px  │   (actual image)   │ px  │
    │     │                    │     │
    └────────────────────────────────┘
              1920 x 1080 ✓
```

### Example 3: Large Square to Standard

```
INPUT:
Method: 1000 x 1000 (square, 1:1 ratio)
Target:  800 x  600 (standard, 4:3 ratio)

STEP 1: Compare aspect ratios
Method ratio: 1000/1000 = 1.0
Target ratio: 800/600 = 1.33
→ Method is TALLER (1.0 < 1.33)

STEP 2: Scale to match height
Scale factor: 600/1000 = 0.6
New size: 600 x 600

STEP 3: Pad horizontally
Need: 800 - 600 = 200px padding
Add: 100px left + 100px right

OUTPUT:
    ┌──────────────────┐
    │   │          │   │
    │100│ 600x600  │100│
    │px │  SQUARE  │px │
    │   │          │   │
    └──────────────────┘
         800 x 600 ✓
```

### Example 4: Slight Mismatch (Cropping)

```
INPUT:
Method: 1024 x 768 (4:3 ratio)
Target:  800 x 600 (4:3 ratio)

STEP 1: Compare aspect ratios
Method ratio: 1024/768 = 1.33
Target ratio: 800/600 = 1.33
→ Same ratio!

STEP 2: Scale to match
Scale factor: 800/1024 = 0.78125
New size: 800 x 600

STEP 3: No adjustment needed
Perfect match!

OUTPUT:
       ┌──────────────┐
       │              │
       │  SCALED IMG  │
       │              │
       └──────────────┘
           800 x 600 ✓
```

---

## Quality Considerations

### Resampling Method: LANCZOS

The program uses Lanczos resampling, which provides the best quality for most use cases:

```
DOWNSCALING (large → small):
┌────────────────┐
│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│  High detail
│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│  Sharp edges
│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│  Smooth gradients
└────────────────┘
        ↓ LANCZOS
    ┌──────┐
    │▓▓▓▓▓▓│  Preserved detail
    │▓▓▓▓▓▓│  Minimal artifacts
    └──────┘

UPSCALING (small → large):
    ┌──────┐
    │▓▓▓▓▓▓│  Original
    └──────┘
        ↓ LANCZOS
┌────────────────┐
│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│  Smooth interpolation
│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│  Good edge quality
└────────────────┘
```

---

## Decision Tree

```
START
  │
  ├─→ Load target and method images
  │
  ├─→ Calculate aspect ratios
  │      │
  │      ├─→ method_ratio > target_ratio?
  │      │   YES → Scale to match WIDTH
  │      │   NO  → Scale to match HEIGHT
  │
  ├─→ Resize using LANCZOS
  │
  ├─→ Check dimensions
  │      │
  │      ├─→ Too small?
  │      │   YES → PAD (add colored pixels)
  │      │         │
  │      │         ├─→ Use pad-color option
  │      │         │   (default: black)
  │      │         └─→ Center the image
  │      │
  │      └─→ Too large?
  │          YES → CROP (remove pixels)
  │                │
  │                ├─→ Use crop-mode option
  │                │   (default: center)
  │                └─→ Keep best portion
  │
  └─→ Save result
      (exact target dimensions)
```

---

## Summary

**Key Principles:**
1. ✅ **Always preserve aspect ratio** during scaling
2. ✅ **Never distort** the image
3. ✅ **High quality** resampling (LANCZOS)
4. ✅ **Exact dimensions** guaranteed
5. ✅ **Configurable** padding and cropping

**The result:** Professional-quality resizing that respects the content of your images while meeting exact dimensional requirements.
