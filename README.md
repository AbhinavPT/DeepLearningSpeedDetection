# Vehicle Speed Detection System

## Overview
This project implements a vehicle speed detection system using computer vision techniques. It can detect vehicles, track their movements, and calculate their speeds within a specified region of interest.
fine-tune-yolo.ipynb contains the code used to fine-tune YOLO for extreme weather conditions, including snow, rain, and fog.

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/AbhinavPT/DeepLearningSpeedDetection.git
cd DeepLearningSpeedDetection
```

### 2. Set Up Virtual Environment

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**Linux/MacOS:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Model Weights
Download the fine-tuned weights from [Google Drive Link](https://drive.google.com/drive/folders/1wUMxLzoCKjBaip6-eLmWi1F8NP-bc2Ig?usp=drive_link) and place the file named `best.pt` in the project root directory.

### 5. Calibration Setup
Before running the system, calibrate the `TARGET_HEIGHT` variable in the code:
- Measure the actual distance (in meters) along the road within your region of interest
- Open `app.py` and set `TARGET_HEIGHT = your_measured_distance`
```python
# Example: If your ROI covers 50 meters along the road
TARGET_HEIGHT = 50  # in meters
```
This calibration is crucial for accurate speed measurements and its 50 by default.

## Usage

### 1. Start the Application
```bash
streamlit run app.py
```

### 2. Upload Video
- Open the displayed URL in your browser
- Click "Upload source video" and select your video file

### 3. Define Region of Interest (ROI)
- An OpenCV window will appear showing the first frame
- ROI is the area where vehicle detection and speed measurement will occur
- Click 4 points in this order to define ROI:
  1. Top-left corner
  2. Top-right corner
  3. Bottom-right corner
  4. Bottom-left corner
- Press 'q' after selecting all points

```
    1 -------- 2
    |          |
    |   ROI    |
    |          |
    4 -------- 3
```

Important: When selecting ROI points, ensure:
- The road section is relatively straight
- The distance between points 1-2 and 4-3 corresponds to your calibrated TARGET_HEIGHT
- The region is clear and unobstructed

### 4. View Results
- Progress bar shows processing status
- Processed video will appear in the browser
- Output video is saved in `./target` folder
- Speed statistics are displayed on the webpage

## Output Details

### Processed Video Features
- Vehicle bounding boxes
- Vehicle type classification
- Real-time speed measurements (km/h)
- Vehicle tracking trajectories

### Speed Analysis
- Total vehicle count
- Minimum Speed (15th percentile)
- Safe Speed (85th percentile)
- Design Speed (98th percentile)

## Project Structure
```
DeepLearningSpeedDetection/
│
├── app.py                 # Main application file
├── requirements.txt       # Package dependencies
├── best.pt               # Model weights
│
└── target/               # Output directory
```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
