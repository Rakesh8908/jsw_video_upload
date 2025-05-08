# Cement Bag Detection with YOLOv8

This project uses YOLOv8 to detect and count moving cement bags in a warehouse environment.

## Dataset Preparation

1. Create the following directory structure:
```
datasets/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

2. Collect and annotate your images:
   - Take videos of moving cement bags in your warehouse
   - Extract frames from the videos
   - Annotate the cement bags in each image using a tool like LabelImg or CVAT
   - Save annotations in YOLO format (class_id x_center y_center width height)
   - Split your dataset into train (70%), validation (20%), and test (10%) sets

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training

1. Place your dataset in the correct directory structure
2. Update the `data.yaml` file if needed
3. Run the training script:
```bash
python train.py
```

The trained model will be saved in the `runs/train/cement_bags/weights` directory.

## Model Usage

After training, you can use the best.pt model to:
- Detect cement bags in new images/videos
- Count moving cement bags in real-time
- Track cement bags across frames

## Notes

- For better results, ensure your training data includes:
  - Various lighting conditions
  - Different angles of cement bags
  - Different distances from the camera
  - Occluded cement bags
  - Moving cement bags 