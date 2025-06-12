import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification, ViTFeatureExtractor, ViTForImageClassification
import matplotlib.pyplot as plt
from pathlib import Path
import time

class VideoAgeDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load multiple age detection models for better accuracy
        self.load_models()
        
        # Initialize advanced face detection (replacing low-accuracy Haar Cascade)
        self.load_face_detectors()
        
        # Age groups mapping for different models
        self.age_groups_siglip = {
            0: "1-10", 1: "11-20", 2: "21-30", 3: "31-40",
            4: "41-55", 5: "56-65", 6: "66-80", 7: "80+"
        }
        
        self.age_groups_vit = {
            0: "0-2", 1: "3-9", 2: "10-19", 3: "20-29", 4: "30-39", 
            5: "40-49", 6: "50-59", 7: "60-69", 8: "70+"
        }
        
    def load_models(self):
        """Load age detection models"""
        print("Loading age detection models...")
        
        try:
            # Load SigLIP-based facial age detection model
            self.siglip_model_name = "prithivMLmods/facial-age-detection"
            self.siglip_model = SiglipForImageClassification.from_pretrained(self.siglip_model_name)
            self.siglip_processor = AutoImageProcessor.from_pretrained(self.siglip_model_name)
            self.siglip_model.to(self.device)
            self.siglip_model.eval()
            print("✓ SigLIP age detection model loaded")
        except Exception as e:
            print(f"Failed to load SigLIP model: {e}")
            self.siglip_model = None
            
        try:
            # Load ViT-based age classifier as backup
            self.vit_model_name = "nateraw/vit-age-classifier"
            self.vit_model = ViTForImageClassification.from_pretrained(self.vit_model_name)
            self.vit_processor = ViTFeatureExtractor.from_pretrained(self.vit_model_name)
            self.vit_model.to(self.device)
            self.vit_model.eval()
            print("✓ ViT age classifier model loaded")
        except Exception as e:
            print(f"Failed to load ViT model: {e}")
            self.vit_model = None
    
    def load_face_detectors(self):
        """Load advanced face detection models - much better than Haar Cascade"""
        print("🔄 Loading advanced face detection models...")
        
        # Import here to avoid issues if not installed
        import mediapipe as mp
        from mtcnn import MTCNN
        
        # Initialize MediaPipe Face Detection (Primary)
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 1 for full range detection
            min_detection_confidence=0.6
        )
        print("✅ MediaPipe Face Detection loaded")
        
        # Initialize MTCNN (Secondary/Backup)
        try:
            self.mtcnn = MTCNN()
            print("✅ MTCNN Face Detection loaded")
        except Exception as e:
            print(f"⚠️ MTCNN loading failed: {e}")
            self.mtcnn = None
    
    def detect_faces_mediapipe(self, frame):
        """Detect faces using MediaPipe - superior to Haar Cascade"""
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
        """Detect faces using MTCNN - excellent backup"""
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
        """Advanced face detection using MediaPipe + MTCNN (replacing Haar Cascade)"""
        # Try MediaPipe first (faster and very accurate)
        faces = self.detect_faces_mediapipe(frame)
        
        # If MediaPipe fails or finds no faces, try MTCNN
        if not faces and self.mtcnn:
            faces = self.detect_faces_mtcnn(frame)
        
        # Sort by confidence if available
        if faces and len(faces[0]) > 4:
            faces = sorted(faces, key=lambda x: x[4], reverse=True)
        
        return faces
    
    def predict_age_siglip(self, face_img):
        """Predict age using SigLIP model"""
        if self.siglip_model is None:
            return None, 0.0
            
        try:
            face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            face_pil = face_pil.resize((224, 224))
            
            inputs = self.siglip_processor(images=face_pil, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.siglip_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                predicted_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0][predicted_class].item()
                
            return self.age_groups_siglip[predicted_class], confidence
        except Exception as e:
            print(f"SigLIP prediction error: {e}")
            return None, 0.0
    
    def predict_age_vit(self, face_img):
        """Predict age using ViT model"""
        if self.vit_model is None:
            return None, 0.0
            
        try:
            face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            face_pil = face_pil.resize((224, 224))
            
            inputs = self.vit_processor(face_pil, return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # Add error handling for memory issues
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                outputs = self.vit_model(**inputs)
                probs = outputs.logits.softmax(1)
                predicted_class = probs.argmax(1).item()
                confidence = probs[0][predicted_class].item()
                
            return self.age_groups_vit[predicted_class], confidence
        except Exception as e:
            print(f"ViT prediction error: {e}")
            return None, 0.0
    
    def get_best_age_prediction(self, face_img):
        """Get the best age prediction from available models"""
        predictions = []
        
        # Try SigLIP model
        age_siglip, conf_siglip = self.predict_age_siglip(face_img)
        if age_siglip:
            predictions.append((age_siglip, conf_siglip, "SigLIP"))
        
        # Try ViT model
        age_vit, conf_vit = self.predict_age_vit(face_img)
        if age_vit:
            predictions.append((age_vit, conf_vit, "ViT"))
        
        if not predictions:
            return "Unknown", 0.0, "None"
        
        # Return the prediction with highest confidence
        best_pred = max(predictions, key=lambda x: x[1])
        return best_pred
    
    def process_video(self, video_path, output_path=None, display_live=True):
        """Process video and detect ages in real-time"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer if output path is provided
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        if output_path:
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect faces
            faces = self.detect_faces(frame)
            
            # Process each detected face
            for i, face_data in enumerate(faces):
                if len(face_data) >= 4:
                    x, y, w, h = face_data[:4]
                    confidence_score = face_data[4] if len(face_data) > 4 else 1.0
                    
                    # Extract face region with some padding
                    padding = 20
                    y1 = max(0, y - padding)
                    y2 = min(height, y + h + padding)
                    x1 = max(0, x - padding)
                    x2 = min(width, x + w + padding)
                    
                    face_img = frame[y1:y2, x1:x2]
                    
                    if face_img.size > 0:
                        # Predict age
                        age_range, confidence, model_used = self.get_best_age_prediction(face_img)
                        
                        # Draw bounding box around face (different colors for different people)
                        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
                        color = colors[i % len(colors)]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                        
                        # Create labels with detection confidence
                        age_label = f"Age: {age_range}"
                        conf_label = f"Age Conf: {confidence:.2f}"
                        det_label = f"Det Conf: {confidence_score:.2f}"
                        
                        # Calculate label dimensions
                        age_size = cv2.getTextSize(age_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        conf_size = cv2.getTextSize(conf_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        
                        # Draw label background
                        label_height = 70
                        cv2.rectangle(frame, (x, y - label_height), 
                                    (x + max(age_size[0], conf_size[0]) + 10, y), color, -1)
                        
                        # Draw labels
                        cv2.putText(frame, age_label, (x + 5, y - 45), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                        cv2.putText(frame, conf_label, (x + 5, y - 25), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        cv2.putText(frame, det_label, (x + 5, y - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        
                        # Add person number
                        person_label = f"Person {i+1}"
                        cv2.putText(frame, person_label, (x, y + h + 20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add frame info
            frame_info = f"Frame: {frame_count}/{total_frames} | Faces: {len(faces)}"
            cv2.putText(frame, frame_info, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add detection method info
            method_text = "Detection: MediaPipe + MTCNN (Advanced)"
            cv2.putText(frame, method_text, (10, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Calculate and display FPS
            elapsed_time = time.time() - start_time
            current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            fps_text = f"FPS: {current_fps:.1f}"
            cv2.putText(frame, fps_text, (10, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame to output video
            if out:
                out.write(frame)
            
            # Display frame
            if display_live:
                cv2.imshow('Age Detection - Press Q to quit', frame)
                
                # Break on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Print progress
            if frame_count % 30 == 0:  # Every 30 frames
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% | FPS: {current_fps:.1f}")
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        if display_live:
            cv2.destroyAllWindows()
        
        print(f"Processing completed! Processed {frame_count} frames in {elapsed_time:.2f}s")
        print(f"Average FPS: {frame_count / elapsed_time:.2f}")

def main():
    # Initialize the detector
    detector = VideoAgeDetector()
    
    # Process the video
    video_path = Path("v1.mp4")
    output_path = Path("output_with_age_detection.mp4")
    
    if not video_path.exists():
        print(f"Error: Video file {video_path} not found!")
        return
    
    print("Starting video age detection...")
    print("Press 'Q' to quit during playback")
    
    try:
        detector.process_video(
            video_path=video_path,
            output_path=output_path,
            display_live=True
        )
        
        print(f"\nOutput video saved as: {output_path}")
        
    except Exception as e:
        print(f"Error processing video: {e}")

if __name__ == "__main__":
    main() 