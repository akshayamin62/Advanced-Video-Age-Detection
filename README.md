# 🎥 Advanced Video Age Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.3.0-green.svg)](https://flask.palletsprojects.com/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-orange.svg)](https://mediapipe.dev/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **AI-powered video age detection system using state-of-the-art face detection and classification models**

![Demo](https://img.shields.io/badge/Demo-Available-brightgreen.svg)

## 🚀 **Features**

### 🎯 **Advanced AI Models**
- **🔬 MediaPipe Face Detection**: Google's state-of-the-art face detection (replacing low-accuracy Haar Cascade)
- **🤖 SigLIP2 Age Classification**: Cutting-edge transformer model for age prediction
- **📊 8 Age Groups**: 1-10, 11-20, 21-30, 31-40, 41-55, 56-65, 66-80, 80+ years
- **🎯 Confidence Scoring**: Real-time confidence metrics for both detection and classification

### 🌐 **Dual Interface Options**
- **🌍 Flask Web App**: Beautiful, responsive web interface for browser-based uploads
- **💻 Command Line Interface**: Direct video processing for advanced users
- **📱 Cross-Platform**: Works on Windows, macOS, and Linux

### ⚡ **Performance Features**
- **🏃‍♂️ Real-time Processing**: Optimized frame-by-frame analysis
- **🎨 Visual Annotations**: Color-coded bounding boxes, age labels, and person numbering
- **📈 Live Statistics**: FPS counter, progress tracking, and processing metrics
- **🔄 Smart Caching**: Optimized face detection intervals for better performance

## 📋 **Requirements**

- **Python**: 3.8 or higher
- **GPU**: Optional (CUDA support for faster processing)
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB for models and dependencies

## 🛠️ **Installation**

### **Method 1: Quick Setup**
```bash
# Clone the repository
git clone https://github.com/akshayamin62/Advanced-Video-Age-Detection.git
cd Advanced-Video-Age-Detection

# Install dependencies
pip install -r requirements.txt
```

### **Method 2: Virtual Environment (Recommended)**
```bash
# Clone and create virtual environment
git clone https://github.com/akshayamin62/Advanced-Video-Age-Detection.git
cd Advanced-Video-Age-Detection

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🎮 **Usage**

### 🌐 **Option 1: Web Interface (Recommended)**

1. **Start the Flask server:**
   ```bash
   python app.py
   ```

2. **Open your browser and go to:**
   ```
   http://localhost:5000
   ```

3. **Upload and process:**
   - Drag & drop your video file (MP4, AVI, MOV, MKV)
   - Click "Process Video"
   - Download the annotated result

### 💻 **Option 2: Command Line Interface**

```bash
# Process video directly
python video_age_detector.py

# The script will look for 'v1.mp4' and output 'output_with_age_detection.mp4'
```

## 📊 **Supported Formats**

| **Input Formats** | **Output Format** |
|-------------------|-------------------|
| MP4, AVI, MOV, MKV | MP4 with annotations |

## 🎯 **Age Classification Groups**

| **Age Group** | **Range** | **Color Code** |
|---------------|-----------|----------------|
| 🍼 Infant | 1-10 years | Green |
| 👶 Child | 11-20 years | Blue |
| 🧑 Young Adult | 21-30 years | Red |
| 👨 Adult | 31-40 years | Yellow |
| 👨‍💼 Middle Age | 41-55 years | Purple |
| 👨‍🦳 Mature | 56-65 years | Cyan |
| 👴 Senior | 66-80 years | Orange |
| 👵 Elderly | 80+ years | Pink |

## 🔧 **Technical Architecture**

### **Face Detection Pipeline**
```
Video Frame → MediaPipe Detection → MTCNN Backup → Face Extraction → Age Classification
```

### **Models Used**
- **Primary Face Detection**: MediaPipe Face Detection (Google)
- **Backup Face Detection**: MTCNN (Multi-task CNN)
- **Age Classification**: `prithivMLmods/facial-age-detection` (SigLIP2)

### **Processing Flow**
1. **Video Input**: Load and analyze video properties
2. **Frame Processing**: Extract frames for analysis
3. **Face Detection**: Locate faces using advanced algorithms
4. **Age Prediction**: Classify age using transformer models
5. **Annotation**: Draw bounding boxes, labels, and confidence scores
6. **Output Generation**: Save annotated video with statistics

## 📸 **Sample Output**

The system generates videos with:
- 🎯 **Bounding Boxes**: Color-coded for each person
- 🏷️ **Age Labels**: Predicted age range with confidence
- 📊 **Detection Confidence**: Face detection accuracy scores
- 👥 **Person Numbering**: Individual identification
- 📈 **Performance Metrics**: FPS, frame count, processing stats

### **Demo Video Sample**
A processed video sample is available in the `outputs/` folder:
- **File**: `processed_1749707176_v3.mp4`
- **Features**: Shows real-time age detection with MediaPipe face detection
- **Annotations**: Color-coded bounding boxes, age predictions, and confidence scores

## 🛠️ **Configuration**

### **Advanced Settings** (modify in code)
```python
# Face detection confidence threshold
min_detection_confidence = 0.6

# Face detection interval (for performance)
face_detection_interval = 2  # Every N frames

# Age prediction confidence display
show_confidence_scores = True
```

## 🚀 **Performance Optimization**

### **Speed Improvements**
- **Smart Detection**: Faces detected every 2 frames (configurable)
- **GPU Acceleration**: Automatic CUDA support when available
- **Memory Management**: Efficient frame processing and cleanup

### **Accuracy Improvements**
- **Dual Model System**: MediaPipe + MTCNN for maximum coverage
- **Confidence Thresholds**: Filtering low-confidence detections
- **Multi-angle Support**: Robust detection across viewing angles

## 🔬 **Model Details**

### **MediaPipe Face Detection**
- **Type**: TensorFlow Lite optimized
- **Accuracy**: >95% in optimal conditions
- **Speed**: ~30-60 FPS on CPU
- **Strengths**: Fast, accurate, handles multiple faces

### **SigLIP2 Age Classification**
- **Architecture**: Vision Transformer (ViT)
- **Training Data**: Large-scale age-diverse dataset
- **Accuracy**: ~85% age group classification
- **Output**: 8 distinct age categories with confidence

## 🐛 **Troubleshooting**

### **Common Issues**

| **Issue** | **Solution** |
|-----------|--------------|
| ModuleNotFoundError | Run `pip install -r requirements.txt` |
| Video not loading | Check file format (MP4, AVI, MOV, MKV) |
| Slow processing | Reduce video resolution or enable GPU |
| No faces detected | Check lighting and face visibility |
| Out of memory | Close other applications, use smaller videos |

### **Performance Tips**
- 🎥 **Video Resolution**: 720p-1080p recommended for best speed/quality balance
- 💾 **Available RAM**: Close unnecessary programs during processing
- 🔥 **GPU Usage**: Install CUDA toolkit for GPU acceleration
- 📁 **File Size**: Larger files take longer but produce better results

## 🤝 **Contributing**

We welcome contributions! Please see our contributing guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Commit changes**: `git commit -am 'Add feature'`
4. **Push to branch**: `git push origin feature-name`
5. **Submit a Pull Request**

### **Areas for Contribution**
- 🔧 Additional age classification models
- 🎨 UI/UX improvements
- ⚡ Performance optimizations
- 📱 Mobile app development
- 🌐 Additional language support

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **Google MediaPipe** - Advanced face detection framework
- **HuggingFace Transformers** - Age classification models
- **MTCNN** - Multi-task CNN for face detection
- **OpenCV** - Computer vision library
- **Flask** - Web framework for Python





---

**⭐ If you found this project helpful, please give it a star on GitHub!**

---

<div align="center">
  <b>Built with ❤️ using Python, MediaPipe, and AI</b>
</div> #   A d v a n c e d - V i d e o - A g e - D e t e c t i o n  
 