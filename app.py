from flask import Flask, request, render_template, send_file, jsonify, redirect, url_for
import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification
import tempfile
import time
from pathlib import Path
from werkzeug.utils import secure_filename
import mediapipe as mp
from mtcnn import MTCNN

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs('static', exist_ok=True)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

class AdvancedFaceDetector:
    """Advanced face detection using MediaPipe and MTCNN - much better than Haar Cascade"""
    
    def __init__(self):
        print("🔄 Initializing advanced face detection models...")
        
        # Initialize MediaPipe Face Detection (Primary - Google's state-of-the-art)
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 1 for full range detection (better for videos)
            min_detection_confidence=0.6
        )
        print("✅ MediaPipe Face Detection loaded")
        
        # Initialize MTCNN (Secondary/Backup - Multi-task CNN)
        try:
            self.mtcnn = MTCNN()
            print("✅ MTCNN Face Detection loaded")
        except Exception as e:
            print(f"⚠️ MTCNN loading failed: {e}")
            self.mtcnn = None
    
    def detect_faces_mediapipe(self, frame):
        """Detect faces using MediaPipe - much more accurate than Haar Cascade"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)
            
            faces = []
            if results.detections:
                h, w, _ = frame.shape
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    x = int(bboxC.xmin * w)
                    y = int(bboxC.ymin * h)
                    width = int(bboxC.width * w)
                    height = int(bboxC.height * h)
                    
                    # Ensure coordinates are within frame bounds
                    x = max(0, x)
                    y = max(0, y)
                    width = min(width, w - x)
                    height = min(height, h - y)
                    
                    if width > 30 and height > 30:
                        confidence = detection.score[0]
                        faces.append((x, y, width, height, confidence))
            
            return faces
        except Exception as e:
            print(f"MediaPipe detection error: {e}")
            return []
    
    def detect_faces_mtcnn(self, frame):
        """Detect faces using MTCNN - excellent backup method"""
        if self.mtcnn is None:
            return []
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = self.mtcnn.detect_faces(rgb_frame)
            
            faces = []
            for detection in detections:
                if detection['confidence'] > 0.7:
                    x, y, width, height = detection['box']
                    x = max(0, x)
                    y = max(0, y)
                    confidence = detection['confidence']
                    faces.append((x, y, width, height, confidence))
            
            return faces
        except Exception as e:
            print(f"MTCNN detection error: {e}")
            return []
    
    def detect_faces(self, frame):
        """Primary face detection - uses best available model"""
        # Try MediaPipe first (faster and very accurate)
        faces = self.detect_faces_mediapipe(frame)
        
        # If MediaPipe fails or finds no faces, try MTCNN
        if not faces and self.mtcnn:
            faces = self.detect_faces_mtcnn(frame)
        
        # Sort by confidence
        if faces and len(faces[0]) > 4:
            faces = sorted(faces, key=lambda x: x[4], reverse=True)
        
        return faces

class VideoAgeDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load age detection model
        self.load_age_model()
        
        # Initialize advanced face detection (replacing Haar Cascade)
        self.face_detector = AdvancedFaceDetector()
        
        # Age groups mapping
        self.age_groups = {
            0: "1-10", 1: "11-20", 2: "21-30", 3: "31-40",
            4: "41-55", 5: "56-65", 6: "66-80", 7: "80+"
        }
        
    def load_age_model(self):
        """Load SigLIP age detection model"""
        print("🔄 Loading age detection model...")
        
        try:
            self.model_name = "prithivMLmods/facial-age-detection"
            self.model = SiglipForImageClassification.from_pretrained(self.model_name)
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            print("✅ Age detection model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load age model: {e}")
            raise e
    
    def predict_age(self, face_img):
        """Predict age using SigLIP model"""
        try:
            face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            face_pil = face_pil.resize((224, 224))
            
            inputs = self.processor(images=face_pil, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                predicted_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0][predicted_class].item()
                
            return self.age_groups[predicted_class], confidence
        except Exception as e:
            print(f"Age prediction error: {e}")
            return "Unknown", 0.0
    
    def process_video(self, input_path, output_path):
        """Process video with advanced face detection and age estimation"""
        cap = cv2.VideoCapture(str(input_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        face_detection_interval = 2  # Detect faces every 2 frames for performance
        cached_faces = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect faces periodically
            if frame_count % face_detection_interval == 0:
                faces = self.face_detector.detect_faces(frame)
                cached_faces = faces
            else:
                faces = cached_faces
            
            # Process each detected face
            for i, face_data in enumerate(faces):
                if len(face_data) >= 4:
                    x, y, w, h = face_data[:4]
                    confidence_score = face_data[4] if len(face_data) > 4 else 1.0
                    
                    # Extract face region with padding
                    padding = 15
                    y1 = max(0, y - padding)
                    y2 = min(height, y + h + padding)
                    x1 = max(0, x - padding)
                    x2 = min(width, x + w + padding)
                    
                    face_img = frame[y1:y2, x1:x2]
                    
                    if face_img.size > 0:
                        # Predict age
                        age_range, age_confidence = self.predict_age(face_img)
                        
                        # Draw bounding box (different colors)
                        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
                        color = colors[i % len(colors)]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                        
                        # Create labels
                        age_label = f"Age: {age_range}"
                        conf_label = f"Conf: {age_confidence:.2f}"
                        det_label = f"Det: {confidence_score:.2f}"
                        
                        # Draw label background
                        label_height = 70
                        cv2.rectangle(frame, (x, y - label_height), (x + 200, y), color, -1)
                        
                        # Draw labels
                        cv2.putText(frame, age_label, (x + 5, y - 45), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                        cv2.putText(frame, conf_label, (x + 5, y - 25), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        cv2.putText(frame, det_label, (x + 5, y - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        
                        # Person number
                        person_label = f"Person {i+1}"
                        cv2.putText(frame, person_label, (x, y + h + 20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add info overlay
            info_text = f"Frame: {frame_count}/{total_frames} | Faces: {len(faces)}"
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            method_text = "Detection: MediaPipe + MTCNN (Advanced)"
            cv2.putText(frame, method_text, (10, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Calculate FPS
            elapsed_time = time.time() - start_time
            current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            fps_text = f"FPS: {current_fps:.1f}"
            cv2.putText(frame, fps_text, (10, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Write frame
            out.write(frame)
            
            # Progress update
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% | FPS: {current_fps:.1f} | Faces: {len(faces)}")
        
        # Cleanup
        cap.release()
        out.release()
        
        processing_time = time.time() - start_time
        avg_fps = frame_count / processing_time
        
        print(f"✅ Processing completed!")
        print(f"📊 Stats: {frame_count} frames in {processing_time:.2f}s (avg {avg_fps:.1f} FPS)")
        
        return {
            'frames_processed': frame_count,
            'processing_time': processing_time,
            'avg_fps': avg_fps,
            'total_frames': total_frames
        }

# Initialize the detector
detector = VideoAgeDetector()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = str(int(time.time()))
            input_filename = f"{timestamp}_{filename}"
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
            file.save(input_path)
            
            # Process video
            output_filename = f"processed_{input_filename}"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            
            print(f"🎥 Processing: {input_filename}")
            stats = detector.process_video(input_path, output_path)
            
            # Clean up input file
            os.remove(input_path)
            
            return jsonify({
                'success': True,
                'output_file': output_filename,
                'stats': stats,
                'download_url': f'/download/{output_filename}'
            })
            
        except Exception as e:
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file format. Use MP4, AVI, MOV, or MKV'}), 400

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "File not found", 404

@app.route('/status')
def status():
    return jsonify({
        'status': 'ready',
        'models_loaded': True,
        'face_detection': 'MediaPipe + MTCNN (Advanced)',
        'age_detection': 'SigLIP2'
    })

if __name__ == '__main__':
    print("🚀 Starting Advanced Age Detection Flask Server...")
    print("📈 Using MediaPipe + MTCNN for superior face detection")
    print("🔗 Access the web interface at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000) 