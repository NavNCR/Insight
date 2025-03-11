import logging
import cv2
import os
import time
import sys
import importlib.util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_install_dependency(package):
    """Check if a package is installed, and try to install it if it's not."""
    if importlib.util.find_spec(package) is None:
        logger.info(f"{package} not found. Installing...")
        try:
            import pip
            pip.main(['install', package])
            return True
        except Exception as e:
            logger.error(f"Failed to install {package}: {str(e)}")
            return False
    return True

def setup_dependencies():
    """Setup all required dependencies for the application."""
    logger.info("Setting up dependencies...")
    
    # List of required packages
    required_packages = ['opencv-python', 'numpy', 'pytesseract', 'pyttsx3', 'SpeechRecognition']
    
    for package in required_packages:
        if not check_install_dependency(package):
            return False
    
    # Check for Tesseract OCR installation
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
    except:
        logger.error("Tesseract OCR not found or not properly configured.")
        logger.info("Please install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki")
        logger.info("After installation, add it to your PATH or set pytesseract.pytesseract.tesseract_cmd")
        return False
    
    return True

class CameraHandler:
    """Handles camera operations using OpenCV instead of PiCamera."""
    
    def __init__(self):
        self.camera = None
        
    def initialize(self):
        """Initialize the camera."""
        try:
            self.camera = cv2.VideoCapture(0)  # Use default camera (index 0)
            
            if not self.camera.isOpened():
                logger.error("Failed to open camera")
                return False
                
            logger.info("Camera initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize camera: {str(e)}")
            return False
    
    def capture_image(self, output_path="capture.jpg"):
        """Capture an image and save it to the specified path."""
        if not self.camera:
            logger.error("Camera not initialized")
            return None
            
        ret, frame = self.camera.read()
        
        if not ret:
            logger.error("Failed to capture image")
            return None
            
        cv2.imwrite(output_path, frame)
        logger.info(f"Image captured and saved to {output_path}")
        return output_path
    
    def release(self):
        """Release the camera resources."""
        if self.camera:
            self.camera.release()
            logger.info("Camera resources released")

class OCRClient:
    """OCR client with wakeword detection for Windows."""
    
    def __init__(self):
        self.camera_handler = CameraHandler()
        
    def initialize(self):
        """Initialize the OCR client."""
        if not setup_dependencies():
            logger.error("Failed to set up required dependencies. Exiting.")
            return False
            
        if not self.camera_handler.initialize():
            logger.error("Cannot continue without camera access")
            return False
            
        # Initialize OCR engine
        try:
            import pytesseract
            logger.info("OCR engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OCR engine: {str(e)}")
            return False
            
        # Initialize text-to-speech engine
        try:
            import pyttsx3
            self.tts_engine = pyttsx3.init()
            logger.info("Text-to-speech engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize text-to-speech engine: {str(e)}")
            return False
            
        # Initialize speech recognition
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            logger.info("Speech recognition initialized")
        except Exception as e:
            logger.error(f"Failed to initialize speech recognition: {str(e)}")
            return False
            
        return True
    
    def process_image(self, image_path):
        """Process the image using OCR and extract text."""
        try:
            import pytesseract
            import cv2
            import numpy as np
            
            # Read the image
            image = cv2.imread(image_path)
            
            # Preprocess the image for better OCR results
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # Extract text using pytesseract
            text = pytesseract.image_to_string(thresh)
            
            if text:
                logger.info(f"Extracted text: {text[:100]}...")
                return text
            else:
                logger.warning("No text detected in the image")
                return None
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return None
    
    def speak_text(self, text):
        """Convert text to speech and play it."""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            return True
        except Exception as e:
            logger.error(f"Error speaking text: {str(e)}")
            return False
    
    def listen_for_wakeword(self, wakeword="insight"):
        """Listen for the wakeword using speech recognition."""
        try:
            import speech_recognition as sr
            
            logger.info("Listening for wakeword...")
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source, timeout=5)
                
            try:
                text = self.recognizer.recognize_google(audio).lower()
                logger.info(f"Recognized: {text}")
                return wakeword in text
            except sr.UnknownValueError:
                return False
            except sr.RequestError:
                logger.error("Could not request results from Google Speech Recognition service")
                return False
        except Exception as e:
            logger.error(f"Error listening for wakeword: {str(e)}")
            return False
    
    def run(self):
        """Run the OCR client with wakeword detection."""
        if not self.initialize():
            return
            
        try:
            logger.info("Starting OCR client with wakeword detection...")
            
            while True:
                # Wait for wakeword
                if self.listen_for_wakeword():
                    logger.info("Wakeword detected! Capturing image...")
                    
                    # Capture image
                    image_path = self.camera_handler.capture_image()
                    if not image_path:
                        continue
                        
                    # Process image and extract text
                    text = self.process_image(image_path)
                    if not text:
                        self.speak_text("No text detected")
                        continue
                        
                    # Speak the extracted text
                    self.speak_text(text)
                
                # Small delay to prevent high CPU usage
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            logger.info("Exiting OCR client...")
        finally:
            # Clean up resources
            self.camera_handler.release()

if __name__ == "__main__":
    client = OCRClient()
    client.run()