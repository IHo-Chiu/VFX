# VFX 2023 Spring

## Team 35

member
* R11922189 邱議禾
* R11921117 王鈞

## How to use

### Run MTB

#### MTB Algotrithm

```
python code/MTB.py data/images.csv
```

### Run HDR Algorithm

#### 1. Debevec HDR Algorithm

```
python code/debevec_hdr_algorithm.py data/images.csv --shifted=True
```

#### 2. Robertson HDR Algorithm

```
python code/robertson_hdr_algorithm.py data/images.csv --shifted=True
```

### Run Tone Mapping

#### 1. Reinhard Tone Mapping Algorithm

```
python code/reinhard_tone_mapping_algorithm.py data/hdr_image_debevec.npy
python code/reinhard_tone_mapping_algorithm.py data/hdr_image_robertson.npy
```
