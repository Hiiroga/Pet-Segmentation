# Dataset Validation Report

**Generated:** 2026-11-01  
**Dataset:** Cats and Dogs Breeds Classification Oxford Dataset  
**Source:** `zippyz/cats-and-dogs-breeds-classification-oxford-dataset` (Kaggle)

---

## 1. Acceptance Gate — PASSED ✅

All critical requirements for proceeding to preprocessing are satisfied.

| Requirement | Status | Detail |
|---|---|---|
| Segmentation masks present | ✅ PASS | 7,390 trimap PNG files found |
| Masks are pixel-level annotations | ✅ PASS | Confirmed: grayscale PNG, values 1/2/3 per pixel |
| Image-mask matching reliable | ✅ PASS | 7,390 matched pairs by stem name, 0 unmatched |
| Annotation labels interpretable | ✅ PASS | 1=Foreground, 2=Background, 3=Boundary |
| Archive complete | ✅ PASS | All files extracted, 0 corrupt images in 200-sample audit |

---

## 2. File Counts

| Category | Count |
|---|---|
| Total images (`.jpg`) | 7,390 |
| Total trimaps (`.png`) | 7,390 |
| Matched image–mask pairs | 7,390 |
| Images without mask | 0 |
| Masks without image | 0 |
| macOS `._*` metadata files (excluded) | 7,390 (excluded from all processing) |

> [!NOTE]
> The Kaggle ZIP was created on macOS and contains Apple Double `._*` metadata entries for every file. These are automatically excluded in all data loading by filtering `f.name.startswith("._")`.

---

## 3. Trimap Annotation Validation

**Sample size:** 200 randomly selected pairs (seed=42)

| Check | Result |
|---|---|
| Valid trimaps (values ⊆ {1,2,3}, contains FG+BG) | **200 / 200** |
| Invalid trimaps | 0 |
| Corrupt trimap files | 0 |
| Corrupt image files | 0 |
| Unique pixel values observed | **{1, 2, 3}** |

### Label Convention (Oxford-IIIT Pet Trimap Standard)

| Pixel Value | Semantic Meaning | Project Mapping |
|---|---|---|
| `1` | Foreground (pet body) | → Binary mask = **1.0** |
| `2` | Background | → Binary mask = **0.0** |
| `3` | Boundary / uncertain | → **Ignored** during training (treated as BG=0.0) |

---

## 4. Dataset Structure

```
data/raw/
├── annotations/
│   └── annotations/
│       ├── list.txt          ← 7,390 samples with class/species/breed metadata
│       ├── trainval.txt      ← Official train+val split (not used; we re-split)
│       ├── test.txt          ← Official test split (not used; we re-split)
│       ├── README            ← Dataset documentation
│       └── trimaps/          ← 7,390 pixel-level segmentation masks (.png)
└── images/
    └── images/               ← 7,390 RGB pet photographs (.jpg)
```

---

## 5. Image Properties

| Property | Observation |
|---|---|
| Format | JPEG |
| Channels | RGB (3-channel) |
| Resolution | Variable (e.g. 600×400, 375×500, 450×313, ...) |
| Preprocessing required | Yes — resize to 256×256 |

---

## 6. Metadata File (`list.txt`)

Columns: `Image CLASS-ID SPECIES BREED-ID`

- **SPECIES**: `1` = Cat, `2` = Dog
- **CLASS-ID**: 1–37 (37 breed classes)
- Naming convention: Uppercase first letter → Cat; Lowercase first letter → Dog

---

## 7. Sample Visualizations

8 sample visualizations saved to `outputs/dataset_samples/`:

| File | Contents |
|---|---|
| `sample_01_basset_hound_130.png` | Original \| Trimap \| Binary mask \| Overlay |
| `sample_02_pug_56.png` | Original \| Trimap \| Binary mask \| Overlay |
| `sample_03_shiba_inu_49.png` | Original \| Trimap \| Binary mask \| Overlay |
| `sample_04_pomeranian_13.png` | Original \| Trimap \| Binary mask \| Overlay |
| `sample_05_great_pyrenees_98.png` | Original \| Trimap \| Binary mask \| Overlay |
| `sample_06_havanese_136.png` | Original \| Trimap \| Binary mask \| Overlay |
| `sample_07_basset_hound_151.png` | Original \| Trimap \| Binary mask \| Overlay |
| `sample_08_saint_bernard_21.png` | Original \| Trimap \| Binary mask \| Overlay |

---

## 8. Dataset Splits Plan

| Split | Proportion | Expected Count |
|---|---|---|
| Train | 70% | ~5,173 |
| Validation | 15% | ~1,109 |
| Test | 15% | ~1,108 |

Split strategy: **Stratified by species (cat / dog)** using `StratifiedShuffleSplit`, seed=42.

---

## 9. Conclusion

> [!IMPORTANT]
> The dataset is **fully verified and ready for preprocessing**. All 7,390 image-mask pairs are valid. Trimap pixel values are exactly {1, 2, 3} as expected by the Oxford-IIIT Pet standard. Proceed to Stage 03 (Preprocessing & Splits).

