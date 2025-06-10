# Real-Time Object Detection with OpenCV and MobileNet SSD

![Object Detection Demo](Img/person%20detection.png)

## 🎯 Overview

This project provides an advanced real-time object detection system using OpenCV and MobileNet SSD. It can detect and track 20 different object classes through your webcam with enhanced features including performance monitoring, recording capabilities, detection logging, and interactive controls.

### ✨ Key Features

- **Real-time Detection**: Detects 20 object classes including person, car, dog, cat, bicycle, etc.
- **Performance Monitoring**: Live FPS tracking and detection time analysis
- **Recording Capabilities**: Save detection sessions as video files
- **Screenshot Capture**: Save individual frames with detections
- **Detection Logging**: Comprehensive JSON logs of all detection sessions
- **Interactive Controls**: Pause, resume, clear statistics, and more
- **Object Counting**: Track maximum counts of each detected object type
- **Visual Statistics**: On-screen display of performance metrics and object counts

### 🎮 Interactive Controls

| Key | Action |
|-----|--------|
| `q` | Quit the application |
| `r` | Toggle video recording |
| `s` | Save screenshot |
| `c` | Clear statistics |
| `p` | Pause/Resume detection |

## 🚀 Quick Start

### Prerequisites

- Python 3.6+
- OpenCV 3.3+ (with DNN module)
- Webcam or camera device

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd Object-Detection
```

2. **Install required dependencies**
```bash
# For macOS
brew install opencv
pip install opencv-python opencv-contrib-python imutils numpy

# For Ubuntu/Debian
sudo apt-get install python3-opencv
pip install opencv-python opencv-contrib-python imutils numpy

# For Windows
pip install opencv-python opencv-contrib-python imutils numpy
```

3. **Download the MobileNet SSD model** (if not included)
```bash
# The caffemodel file should be downloaded separately due to size constraints
# You can download it from the official MobileNet SSD repository
```

### Running the Application

```bash
python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
```

#### Command Line Arguments

- `--prototxt`: Path to the Caffe deploy prototxt file (required)
- `--model`: Path to the pre-trained Caffe model (required)
- `--confidence`: Minimum confidence threshold for detections (default: 0.2)

#### Example with custom confidence:
```bash
python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel --confidence 0.5
```

## 📊 Detected Object Classes

The system can detect the following 20 object classes:

| Objects | | | |
|---------|---------|---------|---------|
| Person | Aeroplane | Bicycle | Bird |
| Boat | Bottle | Bus | Car |
| Cat | Chair | Cow | Dining Table |
| Dog | Horse | Motorbike | Potted Plant |
| Sheep | Sofa | Train | TV Monitor |

## 🎥 Output Features

### 1. Live Video Display
- Real-time bounding boxes around detected objects
- Confidence scores for each detection
- Color-coded labels for different object classes

![Person and Bottle Detection](Img/Person%20and%20bottel%20detection.png)

### 2. Performance Statistics
- Current and average FPS
- Frame count and runtime
- Number of unique objects detected
- Maximum count for each object type

![Terminal Output](Img/Terminal%20output.png)

### 3. File Outputs

#### Screenshots (`detections/`)
- High-quality captures of detection frames
- Timestamped filenames
- Includes all visual overlays

![Cat Detection](Img/cat%20image%20detection.png)

#### Video Recordings (`recordings/`)
- Full detection sessions with overlays
- AVI format with XVID codec
- Toggle recording during runtime

#### Detection Logs (`logs/`)
- Comprehensive JSON logs with:
  - Session information (start/end times, duration)
  - Object detection history
  - Performance metrics
  - Frame-by-frame detection data

## 🏗️ Project Structure

```
Object-Detection/
├── real_time_object_detection.py    # Main application
├── MobileNetSSD_deploy.prototxt.txt # Network architecture
├── MobileNetSSD_deploy.caffemodel   # Pre-trained weights
├── README.md                        # This file
├── presentation_content.md          # Detailed presentation
├── Img/                            # Sample output images
│   ├── person detection.png
│   ├── cat image detection.png
│   ├── Person and bottel detection.png
│   └── Terminal output.png
├── detections/                      # Screenshots (auto-created)
├── recordings/                      # Video recordings (auto-created)
└── logs/                           # Detection logs (auto-created)
```

## 🔧 Technical Details

### Model Architecture
- **Network**: MobileNet SSD (Single Shot MultiBox Detector)
- **Input Size**: 300x300 pixels
- **Framework**: Caffe
- **Optimization**: Designed for real-time mobile and embedded vision applications

### Performance Optimizations
- Efficient blob preprocessing
- Optimized frame resizing
- Smart memory management
- FPS monitoring and averaging

### Detection Pipeline
1. **Frame Capture**: Read frame from video stream
2. **Preprocessing**: Resize and normalize image
3. **Inference**: Run through MobileNet SSD
4. **Post-processing**: Filter by confidence and draw bounding boxes
5. **Display**: Show results with statistics overlay

## 🎨 Customization

### Adjusting Detection Sensitivity
Modify the confidence threshold to control detection sensitivity:
- Lower values (0.1-0.3): More detections, potentially more false positives
- Higher values (0.5-0.8): Fewer but more confident detections

### Changing Colors
The system uses random colors for each class. To customize:
```python
# In the ObjectDetectionTracker.__init__ method
self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))
```

### Adding New Features
The modular design makes it easy to add new features:
- Extend the `ObjectDetectionTracker` class
- Add new methods for additional functionality
- Integrate with the main detection loop

## 🐛 Troubleshooting

### Common Issues

1. **Camera not found**
   ```bash
   # Check available cameras (macOS)
   system_profiler SPCameraDataType
   
   # Check available cameras (Linux)
   ls /dev/video*
   ```

2. **Model file missing**
   - Ensure `MobileNetSSD_deploy.caffemodel` is in the project directory
   - Download from the official MobileNet SSD repository if needed

3. **Low FPS performance**
   - Reduce input resolution
   - Increase confidence threshold
   - Close other applications using the camera

4. **OpenCV DNN module not found**
   ```bash
   # Reinstall OpenCV with contrib modules
   pip uninstall opencv-python
   pip install opencv-contrib-python
   ```

## 📈 Performance Benchmarks

Typical performance on different hardware:

| Hardware | Resolution | FPS | Notes |
|----------|------------|-----|-------|
| MacBook Pro M1 | 800x600 | 25-30 | Excellent performance |
| Intel i7 Laptop | 800x600 | 15-20 | Good performance |
| Raspberry Pi 4 | 640x480 | 5-10 | Acceptable for demos |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **MobileNet SSD**: Original paper and implementation by Google
- **OpenCV**: Computer vision library
- **PyImageSearch**: Tutorials and inspiration for object detection
- **Caffe**: Deep learning framework

## 📚 References

- [MobileNet SSD Paper](https://arxiv.org/abs/1512.02325)
- [OpenCV DNN Module](https://docs.opencv.org/master/d2/d58/tutorial_table_of_content_dnn.html)
- [MobileNet SSD GitHub](https://github.com/chuanqi305/MobileNet-SSD)
- [PyImageSearch Object Detection](https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/)

---

**Made with ❤️ for the computer vision community**

## 📸 Sample Detection Results

The system successfully detects various objects in real-time scenarios:

### Person Detection
![Person Detection Example](Img/person%20detection.png)
*Real-time person detection with confidence scores and bounding boxes*

### Multi-Object Detection
![Person and Bottle Detection](Img/Person%20and%20bottel%20detection.png)
*Simultaneous detection of multiple object classes (person and bottle)*

### Animal Detection
![Cat Detection Example](Img/cat%20image%20detection.png)
*Accurate detection of animals with high confidence scores*

### System Performance
![Terminal Output](Img/Terminal%20output.png)
*Console output showing system performance metrics and detection statistics*