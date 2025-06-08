# Enhanced Real-time Object Detection with MobileNet SSD
# How to run: python enhanced_real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

import os
import json
import time
import argparse
import cv2
import numpy as np
import imutils
from imutils.video import VideoStream, FPS
from datetime import datetime
from collections import defaultdict, deque

class ObjectDetectionTracker:
    def __init__(self, args):
        self.args = args
        self.detection_history = defaultdict(list)
        self.object_counts = defaultdict(int)
        self.frame_count = 0
        self.start_time = time.time()
        self.detection_log = []
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.detection_times = deque(maxlen=30)
        
        # Recording setup
        self.recording = False
        self.video_writer = None
        
        # Initialize classes and colors
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                       "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                       "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                       "sofa", "train", "tvmonitor"]
        
        self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))
        
        # Load model
        print("[INFO] Loading model...")
        self.net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
        
        # Initialize video stream
        print("[INFO] Starting video stream...")
        self.vs = VideoStream(src=0).start()
        time.sleep(2.0)
        
        # Initialize FPS counter
        self.fps = FPS().start()
        
        # Create output directories
        self.create_output_directories()
    
    def create_output_directories(self):
        """Create directories for saving outputs"""
        os.makedirs("detections", exist_ok=True)
        os.makedirs("recordings", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
    
    def detect_objects(self, frame):
        """Perform object detection on a frame"""
        detection_start = time.time()
        
        (h, w) = frame.shape[:2]
        resized_image = cv2.resize(frame, (300, 300))
        
        blob = cv2.dnn.blobFromImage(resized_image, (1/127.5), (300, 300), 127.5, swapRB=True)
        self.net.setInput(blob)
        predictions = self.net.forward()
        
        detection_time = time.time() - detection_start
        self.detection_times.append(detection_time)
        
        detections = []
        frame_objects = defaultdict(int)
        
        for i in np.arange(0, predictions.shape[2]):
            confidence = predictions[0, 0, i, 2]
            
            if confidence > self.args["confidence"]:
                idx = int(predictions[0, 0, i, 1])
                if idx >= len(self.CLASSES):
                    continue
                    
                box = predictions[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure bounding box is within frame
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                
                class_name = self.CLASSES[idx]
                frame_objects[class_name] += 1
                
                detection = {
                    'class': class_name,
                    'confidence': float(confidence),
                    'bbox': (startX, startY, endX, endY),
                    'timestamp': datetime.now().isoformat()
                }
                detections.append(detection)
                
                # Update detection history
                self.detection_history[class_name].append({
                    'frame': self.frame_count,
                    'confidence': float(confidence),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Update object counts
        for obj_class, count in frame_objects.items():
            self.object_counts[obj_class] = max(self.object_counts[obj_class], count)
        
        return detections
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on frame"""
        for detection in detections:
            class_name = detection['class']
            confidence = detection['confidence']
            (startX, startY, endX, endY) = detection['bbox']
            
            # Get class index for color
            idx = self.CLASSES.index(class_name)
            color = self.COLORS[idx]
            
            # Draw bounding box
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            
            # Create label with confidence
            label = f"{class_name}: {confidence * 100:.1f}%"
            
            # Calculate label position
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            y = startY - 15 if startY - 15 > 15 else startY + 15
            
            # Draw label background
            cv2.rectangle(frame, (startX, y - label_size[1] - 10), 
                         (startX + label_size[0], y), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (startX, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def draw_statistics(self, frame):
        """Draw performance statistics on frame"""
        # Calculate current FPS
        current_fps = 1.0 / self.detection_times[-1] if self.detection_times else 0
        self.fps_history.append(current_fps)
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        
        # Runtime
        runtime = time.time() - self.start_time
        
        # Statistics text
        stats = [
            f"Frame: {self.frame_count}",
            f"FPS: {avg_fps:.1f}",
            f"Runtime: {runtime:.1f}s",
            f"Objects detected: {len(self.object_counts)}",
        ]
        
        # Draw statistics background
        stats_height = len(stats) * 25 + 10
        cv2.rectangle(frame, (10, 10), (300, stats_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, stats_height), (255, 255, 255), 2)
        
        # Draw statistics text
        for i, stat in enumerate(stats):
            y = 30 + i * 25
            cv2.putText(frame, stat, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw object counts
        if self.object_counts:
            counts_y_start = stats_height + 30
            cv2.putText(frame, "Object Counts:", (15, counts_y_start), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            for i, (obj_class, count) in enumerate(self.object_counts.items()):
                y = counts_y_start + 25 + i * 20
                cv2.putText(frame, f"{obj_class}: {count}", (15, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame
    
    def save_detection_log(self):
        """Save detection log to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/detection_log_{timestamp}.json"
        
        log_data = {
            'session_info': {
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_frames': self.frame_count,
                'total_runtime': time.time() - self.start_time
            },
            'object_counts': dict(self.object_counts),
            'detection_history': dict(self.detection_history),
            'performance': {
                'avg_fps': np.mean(self.fps_history) if self.fps_history else 0,
                'avg_detection_time': np.mean(self.detection_times) if self.detection_times else 0
            }
        }
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"[INFO] Detection log saved to {log_file}")
    
    def toggle_recording(self, frame):
        """Toggle video recording"""
        if not self.recording:
            # Start recording
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recordings/detection_recording_{timestamp}.avi"
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, 
                                              (frame.shape[1], frame.shape[0]))
            self.recording = True
            print(f"[INFO] Started recording to {filename}")
        else:
            # Stop recording
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.recording = False
            print("[INFO] Stopped recording")
    
    def save_screenshot(self, frame):
        """Save current frame as screenshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detections/screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"[INFO] Screenshot saved to {filename}")
    
    def run(self):
        """Main detection loop"""
        print("[INFO] Starting object detection...")
        print("[INFO] Controls:")
        print("  'q' - Quit")
        print("  'r' - Toggle recording")
        print("  's' - Save screenshot")
        print("  'c' - Clear statistics")
        print("  'p' - Pause/Resume")
        
        paused = False
        
        try:
            while True:
                if not paused:
                    frame = self.vs.read()
                    if frame is None:
                        break
                    
                    frame = imutils.resize(frame, width=800)
                    self.frame_count += 1
                    
                    # Detect objects
                    detections = self.detect_objects(frame)
                    
                    # Draw detections and statistics
                    frame = self.draw_detections(frame, detections)
                    frame = self.draw_statistics(frame)
                    
                    # Add recording indicator
                    if self.recording:
                        cv2.circle(frame, (frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)
                        cv2.putText(frame, "REC", (frame.shape[1] - 60, 35), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Add pause indicator
                    if paused:
                        cv2.putText(frame, "PAUSED", (frame.shape[1] // 2 - 50, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                    
                    # Write frame to video if recording
                    if self.recording and self.video_writer:
                        self.video_writer.write(frame)
                    
                    # Update FPS counter
                    self.fps.update()
                
                # Show frame
                cv2.imshow("Enhanced Object Detection", frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    self.toggle_recording(frame)
                elif key == ord("s"):
                    self.save_screenshot(frame)
                elif key == ord("c"):
                    self.object_counts.clear()
                    self.detection_history.clear()
                    print("[INFO] Statistics cleared")
                elif key == ord("p"):
                    paused = not paused
                    print(f"[INFO] {'Paused' if paused else 'Resumed'}")
        
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("[INFO] Cleaning up...")
        
        # Stop FPS counter
        self.fps.stop()
        
        # Stop recording if active
        if self.recording and self.video_writer:
            self.video_writer.release()
        
        # Save detection log
        self.save_detection_log()
        
        # Display final statistics
        print(f"[INFO] Session Summary:")
        print(f"  Total frames processed: {self.frame_count}")
        print(f"  Total runtime: {time.time() - self.start_time:.2f} seconds")
        print(f"  Average FPS: {self.fps.fps():.2f}")
        print(f"  Unique objects detected: {len(self.object_counts)}")
        
        if self.object_counts:
            print("  Object counts:")
            for obj_class, count in self.object_counts.items():
                print(f"    {obj_class}: {count}")
        
        # Cleanup OpenCV
        cv2.destroyAllWindows()
        self.vs.stop()

def main():
    # Construct argument parser
    ap = argparse.ArgumentParser(description="Enhanced Real-time Object Detection")
    ap.add_argument("-p", "--prototxt", required=True,
                    help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,
                    help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.2,
                    help="minimum probability to filter weak predictions")
    
    args = vars(ap.parse_args())
    
    # Validate input files
    if not os.path.exists(args["prototxt"]):
        print(f"[ERROR] Prototxt file not found: {args['prototxt']}")
        return
    
    if not os.path.exists(args["model"]):
        print(f"[ERROR] Model file not found: {args['model']}")
        print("[INFO] You need to download the MobileNetSSD_deploy.caffemodel file")
        return
    
    # Create and run detector
    detector = ObjectDetectionTracker(args)
    detector.run()

if __name__ == "__main__":
    main()
