# Kdenlive Auto-Masking Backend: MobileSAM + LiVOS (OpenVINO)

This is a custom script designed to replace Kdenlive's default SAM 2 auto-masking backend. Instead of running a heavy PyTorch SAM 2 model for video propagation, this script uses **MobileSAM** for initial frame selection, and **LiVOS** for getting a pretty reasonable mask.

## Disclaimer
* I built and run this entirely on Linux (Fedora). Theoretically, since PyTorch and OpenVINO are cross-platform, this might work on Windows or macOS, but I haven't actually tried it. If you run into issues on other operating systems, you might need to do some Googling cause... I don't think I'll be of any help really.
* About updates, I dont know if I can update the script when Kdenlive updates and stuff breaks, the main idea with this was to make a base idea, not really planning on maintaining it really professionally. If it starts breaking for me, I'll push an update (Ideally it shouldn't break but ideals are ideal for a reason)

## Prerequisites
* **Python 3.10** You **NEED** Python 3.10 cause it didn't work on other versions for me.

## Installation

You need to set up a virtual environment and install the specific CPU versions of PyTorch and OpenVINO, followed by the LiVOS repository itself.

**1. Create and activate a Python 3.10 virtual environment:**
```bash
python3.10 -m venv venv
source venv/bin/activate
```

**2. Install PyTorch (CPU version):**
```bash
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cpu
```

**3. Install OpenVINO, MobileSAM, and Timm:**
```bash
pip install openvino
pip install timm
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```

**4. Clone and install LiVOS:**
```bash
git clone https://github.com/uncbiag/LiVOS
cd LiVOS
pip install -e .
```

**5. Download LiVOS weights:**
Run the included download script from inside the LiVOS directory you just cloned. This will automatically create a `./weights` folder and download the models into it.
```bash
python ./download.py
```

**6. Navigate to your Kdenlive installation's automask folder and replace the sam-objectmask.py script with the one from this repository (It was in `/usr/share/kdenlive/scripts/automask` for me)**
```text
path/to/scripts/automask/
├── sam-objectmask.py          # The updated python script
├── mobile_sam.pt              # MobileSAM weights (auto downloads)
└── weights/
    └── livos-nomose-480p.pth  # Downloaded LiVOS weights (youll have to copy them after download.py)
```

**7. Point your Kdenlive's "Object Detection" plugin to your venv's python binary:**
You'll also need to download the sam2-tiny model cause kdenlive wont let you use the plugin if there's no SAM model installed
