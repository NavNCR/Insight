#!/usr/bin/env python3

import os
import io
import time
import signal
import logging
import subprocess
import tempfile
import threading
import requests
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ocr_client.log")
    ]
)
logger = logging.getLogger("OCR_Client")

# OCR Configuration
OCR_SERVER_URL = "http://your-server-ip:5000/ocr"
# Lower resolution for faster processing (960x540 - quarter of Full HD)
CAMERA_RESOLUTION = (960, 540)
# Camera warmup time in seconds
CAMERA_WARMUP_TIME = 1

# Wakeword Configuration
ACCESS_KEY = "APy3GFzbJMwPAskDcUB4qJQKIxdc9W0u5ENvlLdhFKm/98Flrmaj5g=="
KEYWORD_PATH = "wakeword.ppn"
SENSITIVITY = 0.5

# Global flags
use_pyaudio = True     # Will switch to False if PyAudio is unavailable
is_processing = False  # Prevents multiple simultaneous processing requests

def setup_dependencies():
    """Check and install required dependencies"""
    global use_pyaudio
    
    logger.info("Setting up dependencies...")
    
    # Check for PiCamera
    try:
        from picamera2 import Picamera2 
        logger.info("PiCamera is available")
    except ImportError:
        logger.error("PiCamera not found. Installing...")
        try:
            subprocess.run(['sudo', 'apt', 'install', '-y', 'python3-picamera2'], check=True)
            import picamera
            logger.info("PiCamera installed successfully")
        except Exception as e:
            logger.error(f"Failed to install PiCamera: {e}")
            logger.error("Cannot continue without camera access")
            return False
    
    # Check for wakeword dependencies
    try:
        import pvporcupine
        from pvrecorder import PvRecorder
        logger.info("Wakeword detection libraries available")
    except ImportError:
        logger.error("Wakeword libraries not found. Installing...")
        try:
            subprocess.run(['pip3', 'install', 'pvporcupine pvrecorder'], check=True)
            import pvporcupine
            from pvrecorder import PvRecorder
            logger.info("Wakeword libraries installed successfully")
        except Exception as e:
            logger.error(f"Failed to install wakeword libraries: {e}")
            logger.error("Cannot continue without wakeword detection")
            return False
    
    # Try to import PyAudio (for direct streaming)
    try:
        import pyaudio
        from pydub import AudioSegment
        logger.info("PyAudio and pydub available for audio streaming")
    except ImportError:
        logger.warning("PyAudio or pydub not available. Installing...")
        try:
            # First install system dependencies
            subprocess.run(['sudo', 'apt-get', 'update'], check=True)
            subprocess.run(['sudo', 'apt-get', 'install', '-y', 'portaudio19-dev'], check=True)
            subprocess.run(['pip3', 'install', 'pyaudio', 'pydub'], check=True)
            
            # Try importing again
            import pyaudio
            from pydub import AudioSegment
            logger.info("PyAudio and pydub installed successfully")
        except Exception as e:
            logger.warning(f"Failed to install PyAudio: {e}")
            logger.warning("Will fall back to mpg123 for audio playback")
            
            # Check if mpg123 is installed
            try:
                subprocess.run(['which', 'mpg123'], check=True)
                logger.info("mpg123 is available for fallback playback")
            except:
                logger.warning("Installing mpg123 for fallback playback...")
                try:
                    subprocess.run(['sudo', 'apt-get', 'install', '-y', 'mpg123'], check=True)
                    logger.info("mpg123 installed successfully")
                except Exception as e:
                    logger.error(f"Failed to install mpg123: {e}")
                    logger.error("No audio playback method available")
                    return False
            
            use_pyaudio = False
    
    return True

def capture_image():
    """Capture an image from Raspberry Pi CSI camera with reduced resolution"""
    logger.info("Capturing image from camera...")
    
    try:
        import picamera
        # Create a memory stream
        image_stream = io.BytesIO()
        
        # Initialize the camera with reduced resolution
        with picamera.PiCamera() as camera:
            # Set reduced resolution
            camera.resolution = CAMERA_RESOLUTION
            # Allow camera to warm up, but with reduced time
            time.sleep(CAMERA_WARMUP_TIME)
            # Capture the image directly to memory
            camera.capture(image_stream, format='jpeg')
        
        # Reset stream position
        image_stream.seek(0)
        logger.info("Image captured successfully")
        return image_stream
    except Exception as e:
        logger.error(f"Error capturing image: {e}")
        return None

def send_to_ocr_server(image_stream, server_url=OCR_SERVER_URL):
    """Send image to OCR server and get audio response"""
    logger.info(f"Sending image to OCR server at {server_url}...")
    
    if not image_stream:
        logger.error("No image data to send")
        return None
    
    # Prepare the files for the POST request
    files = {'image': ('image.jpg', image_stream, 'image/jpeg')}
    
    # Send the request with a 30 second timeout
    try:
        response = requests.post(server_url, files=files, timeout=30, stream=True)
        
        if response.status_code == 200:
            logger.info("Received audio response from server")
            return response.content
        else:
            logger.error(f"Error: Server returned status code {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error communicating with server: {e}")
        return None

def play_audio_with_pyaudio(audio_data):
    """Play audio directly from memory using PyAudio and pydub"""
    try:
        import pyaudio
        from pydub import AudioSegment
        from pydub.utils import make_chunks
        
        # Use pydub to load the MP3 data
        audio = AudioSegment.from_mp3(io.BytesIO(audio_data))
        
        # Get audio parameters
        sample_width = audio.sample_width
        channels = audio.channels
        frame_rate = audio.frame_rate
        
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        # Create a stream
        stream = p.open(
            format=p.get_format_from_width(sample_width),
            channels=channels,
            rate=frame_rate,
            output=True
        )
        
        # Use smaller chunks for lower latency
        chunk_size = 512
        buffer = io.BytesIO(audio._data)
        
        logger.info("Starting audio playback with PyAudio...")
        
        # Read and play in small chunks
        data = buffer.read(chunk_size)
        while data:
            stream.write(data)
            data = buffer.read(chunk_size)
        
        # Clean up
        stream.stop_stream()
        stream.close()
        p.terminate()
        logger.info("Audio playback complete")
        return True
    except Exception as e:
        logger.error(f"Error playing audio with PyAudio: {e}")
        return False

def play_audio_with_mpg123(audio_data):
    """Play audio using mpg123 command-line player as fallback"""
    try:
        # Create a temporary file that will be automatically deleted
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(audio_data)
        
        logger.info("Starting audio playback with mpg123...")
        
        # Use mpg123 for playback
        subprocess.run(['mpg123', '-q', temp_path], check=True)
        
        # Remove the temporary file
        os.remove(temp_path)
        
        logger.info("Audio playback complete")
        return True
    except Exception as e:
        logger.error(f"Error playing audio with mpg123: {e}")
        return False

def play_audio_data(audio_data):
    """Plays the received audio data from the OCR server."""
    if not audio_data:
        logger.error("No audio data received for playback.")
        return False

    try:
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_audio.write(audio_data)
        temp_audio.close()

        if use_pyaudio:
            from pydub import AudioSegment
            from pydub.playback import play
            sound = AudioSegment.from_wav(temp_audio.name)
            play(sound)
        else:
            subprocess.run(["mpg123", temp_audio.name])

        os.remove(temp_audio.name)
        return True
    except Exception as e:
        logger.error(f"Error playing audio: {e}")
        return False


def process_wakeword_detected():

    def wakeword_listener():
        """Continuously listens for the wakeword and triggers OCR processing."""
        try:
            porcupine = pvporcupine.create(
                access_key=ACCESS_KEY,
                keyword_paths=[KEYWORD_PATH],
                sensitivities=[SENSITIVITY]
            )

            recorder = PvRecorder(device_index=-1, frame_length=porcupine.frame_length)
            recorder.start()

            logger.info("Wakeword detection started... Say the wakeword!")

            while True:
                pcm = recorder.read()
                keyword_index = porcupine.process(pcm)

                if keyword_index >= 0:
                    logger.info("Wakeword detected! Capturing image...")
                    process_wakeword_detected()
    
        except Exception as e:
            logger.error(f"Wakeword detection error: {e}")
        finally:
            recorder.stop()
            recorder.delete()
            porcupine.delete()


    """Handle the full OCR process when wakeword is detected"""
    global is_processing
    
    # Prevent multiple simultaneous processing requests
    if is_processing:
        logger.warning("Already processing a request, ignoring...")
        return
    
    try:
        is_processing = True
        
        # 1. Capture an image
        image_stream = capture_image()
        if not image_stream:
            logger.error("Failed to capture image, aborting process")
            return
        
        # 2. Send to OCR server
        audio_data = send_to_ocr_server(image_stream)
        if not audio_data:
            logger.error("Failed to get audio response, aborting process")
            return
        
        # 3. Play the response
        play_audio_data(audio_data)
        
        logger.info("OCR process completed successfully")
    except Exception as e:
        logger.error(f"Error in processing pipeline: {e}")
    finally:
        is_processing = False
        logger.info("Returning to wakeword detection...")

def main():
    """Main function to run the wakeword detection and OCR client"""
    logger.info("Starting OCR client with wakeword detection...")

    # Make sure all dependencies are available
    if not setup_dependencies():
        logger.error("Failed to set up required dependencies. Exiting.")
        return

    # Start the wakeword listener in a separate thread
    listener_thread = threading.Thread(target=wakeword_listener, daemon=True)
    listener_thread.start()

    try:
        while True:
            time.sleep(1)  # Keep the main script running
    except KeyboardInterrupt:
        logger.info("Stopping wakeword detection...")


if __name__ == "__main__":
    main()


# import os
# import io
# import time
# import signal
# import logging
# import subprocess
# import tempfile
# import threading
# import requests
# from datetime import datetime

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler(),
#         logging.FileHandler("ocr_client.log")
#     ]
# )
# logger = logging.getLogger("OCR_Client")

# # OCR Configuration
# OCR_SERVER_URL = "http://your-server-ip:5000/ocr"
# # Lower resolution for faster processing (960x540 - quarter of Full HD)
# CAMERA_RESOLUTION = (960, 540)
# # Camera warmup time in seconds
# CAMERA_WARMUP_TIME = 1

# # Wakeword Configuration - "insight" is already our wake word
# ACCESS_KEY = "APy3GFzbJMwPAskDcUB4qJQKIxdc9W0u5ENvlLdhFKm/98Flrmaj5g=="
# KEYWORD_PATH = "wakeword.ppn"  # Assuming this is already set to "insight"
# SENSITIVITY = 0.5

# # Global flags
# use_pyaudio = True     # Will switch to False if PyAudio is unavailable
# is_processing = False  # Prevents multiple simultaneous processing requests

# def setup_dependencies():
#     """Check and install required dependencies"""
#     global use_pyaudio
    
#     logger.info("Setting up dependencies...")
    
#     # Check for PiCamera
#     try:
#         import picamera
#         logger.info("PiCamera is available")
#     except ImportError:
#         logger.error("PiCamera not found. Installing...")
#         try:
#             subprocess.run(['pip3', 'install', 'picamera'], check=True)
#             import picamera
#             logger.info("PiCamera installed successfully")
#         except Exception as e:
#             logger.error(f"Failed to install PiCamera: {e}")
#             logger.error("Cannot continue without camera access")
#             return False
    
#     # Check for wakeword dependencies
#     try:
#         import pvporcupine
#         from pvrecorder import PvRecorder
#         logger.info("Wakeword detection libraries available")
#     except ImportError:
#         logger.error("Wakeword libraries not found. Installing...")
#         try:
#             subprocess.run(['pip3', 'install', 'pvporcupine pvrecorder'], check=True)
#             import pvporcupine
#             from pvrecorder import PvRecorder
#             logger.info("Wakeword libraries installed successfully")
#         except Exception as e:
#             logger.error(f"Failed to install wakeword libraries: {e}")
#             logger.error("Cannot continue without wakeword detection")
#             return False
    
#     # Try to import PyAudio (for direct streaming)
#     try:
#         import pyaudio
#         from pydub import AudioSegment
#         logger.info("PyAudio and pydub available for audio streaming")
#     except ImportError:
#         logger.warning("PyAudio or pydub not available. Installing...")
#         try:
#             # First install system dependencies
#             subprocess.run(['sudo', 'apt-get', 'update'], check=True)
#             subprocess.run(['sudo', 'apt-get', 'install', '-y', 'portaudio19-dev'], check=True)
#             subprocess.run(['pip3', 'install', 'pyaudio', 'pydub'], check=True)
            
#             # Try importing again
#             import pyaudio
#             from pydub import AudioSegment
#             logger.info("PyAudio and pydub installed successfully")
#         except Exception as e:
#             logger.warning(f"Failed to install PyAudio: {e}")
#             logger.warning("Will fall back to mpg123 for audio playback")
            
#             # Check if mpg123 is installed
#             try:
#                 subprocess.run(['which', 'mpg123'], check=True)
#                 logger.info("mpg123 is available for fallback playback")
#             except:
#                 logger.warning("Installing mpg123 for fallback playback...")
#                 try:
#                     subprocess.run(['sudo', 'apt-get', 'install', '-y', 'mpg123'], check=True)
#                     logger.info("mpg123 installed successfully")
#                 except Exception as e:
#                     logger.error(f"Failed to install mpg123: {e}")
#                     logger.error("No audio playback method available")
#                     return False
            
#             use_pyaudio = False
    
#     return True

# def capture_image():
#     """Capture an image from Raspberry Pi CSI camera with reduced resolution"""
#     logger.info("Capturing image from camera...")
    
#     try:
#         import picamera
#         # Create a memory stream
#         image_stream = io.BytesIO()
        
#         # Initialize the camera with reduced resolution
#         with picamera.PiCamera() as camera:
#             # Set reduced resolution
#             camera.resolution = CAMERA_RESOLUTION
#             # Allow camera to warm up, but with reduced time
#             time.sleep(CAMERA_WARMUP_TIME)
#             # Capture the image directly to memory
#             camera.capture(image_stream, format='jpeg')
        
#         # Reset stream position
#         image_stream.seek(0)
#         logger.info("Image captured successfully")
#         return image_stream
#     except Exception as e:
#         logger.error(f"Error capturing image: {e}")
#         return None

# def send_to_ocr_server(image_stream, server_url=OCR_SERVER_URL):
#     """Send image to OCR server and get audio response"""
#     logger.info(f"Sending image to OCR server at {server_url}...")
    
#     if not image_stream:
#         logger.error("No image data to send")
#         return None
    
#     # Prepare the files for the POST request
#     files = {'image': ('image.jpg', image_stream, 'image/jpeg')}
    
#     # Send the request with a 30 second timeout
#     try:
#         response = requests.post(server_url, files=files, timeout=30, stream=True)
        
#         if response.status_code == 200:
#             logger.info("Received audio response from server")
#             return response.content
#         else:
#             logger.error(f"Error: Server returned status code {response.status_code}")
#             return None
#     except requests.exceptions.RequestException as e:
#         logger.error(f"Error communicating with server: {e}")
#         return None

# def play_audio_with_pyaudio(audio_data):
#     """Play audio directly from memory using PyAudio and pydub"""
#     try:
#         import pyaudio
#         from pydub import AudioSegment
#         from pydub.utils import make_chunks
        
#         # Use pydub to load the MP3 data
#         audio = AudioSegment.from_mp3(io.BytesIO(audio_data))
        
#         # Get audio parameters
#         sample_width = audio.sample_width
#         channels = audio.channels
#         frame_rate = audio.frame_rate
        
#         # Initialize PyAudio
#         p = pyaudio.PyAudio()
        
#         # Create a stream
#         stream = p.open(
#             format=p.get_format_from_width(sample_width),
#             channels=channels,
#             rate=frame_rate,
#             output=True
#         )
        
#         # Use smaller chunks for lower latency
#         chunk_size = 512
#         buffer = io.BytesIO(audio._data)
        
#         logger.info("Starting audio playback with PyAudio...")
        
#         # Read and play in small chunks
#         data = buffer.read(chunk_size)
#         while data:
#             stream.write(data)
#             data = buffer.read(chunk_size)
        
#         # Clean up
#         stream.stop_stream()
#         stream.close()
#         p.terminate()
#         logger.info("Audio playback complete")
#         return True
#     except Exception as e:
#         logger.error(f"Error playing audio with PyAudio: {e}")
#         return False

# def play_audio_with_mpg123(audio_data):
#     """Play audio using mpg123 command-line player as fallback"""
#     try:
#         # Create a temporary file that will be automatically deleted
#         with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
#             temp_path = temp_file.name
#             temp_file.write(audio_data)
        
#         logger.info("Starting audio playback with mpg123...")
        
#         # Use mpg123 for playback
#         subprocess.run(['mpg123', '-q', temp_path], check=True)
        
#         # Remove the temporary file
#         os.remove(temp_path)
        
#         logger.info("Audio playback complete")
#         return True
#     except Exception as e:
#         logger.error(f"Error playing audio with mpg123: {e}")
#         return False

# def play_audio_data(audio_data):
#     """Play audio using the best available method"""
#     if not audio_data:
#         logger.error("No audio data to play")
#         return False
        
#     # Try PyAudio first if available
#     if use_pyaudio:
#         success = play_audio_with_pyaudio(audio_data)
#         if success:
#             return True
#         else:
#             logger.warning("Falling back to mpg123 player")
    
#     # Fall back to mpg123
#     return play_audio_with_mpg123(audio_data)

# def play_acknowledgment_sound():
#     """Play a quick sound to acknowledge wake word detection"""
#     try:
#         # Create a simple beep sound using PyAudio if available
#         if use_pyaudio:
#             import pyaudio
#             import numpy as np
            
#             p = pyaudio.PyAudio()
#             volume = 0.5     # range [0.0, 1.0]
#             fs = 44100       # sampling rate, Hz
#             duration = 0.2   # seconds
#             f = 880.0        # sine frequency, Hz
            
#             # Generate samples
#             samples = (np.sin(2 * np.pi * np.arange(fs * duration) * f / fs) * volume * 32767).astype(np.int16)
            
#             # Play the samples
#             stream = p.open(format=p.get_format_from_width(2),
#                             channels=1,
#                             rate=fs,
#                             output=True)
            
#             # Play in chunks for better responsiveness
#             chunk_size = 1024
#             for i in range(0, len(samples), chunk_size):
#                 stream.write(samples[i:i+chunk_size].tobytes())
            
#             stream.stop_stream()
#             stream.close()
#             p.terminate()
#             return True
#         else:
#             # Use a simpler method if PyAudio is not available
#             logger.info("Acknowledged 'insight' wake word")
#             return True
#     except Exception as e:
#         logger.error(f"Error playing acknowledgment sound: {e}")
#         return False

# def process_wakeword_detected():
#     """Handle the full OCR process when 'insight' wake word is detected"""
#     global is_processing
    
#     # Prevent multiple simultaneous processing requests
#     if is_processing:
#         logger.warning("Already processing a request, ignoring...")
#         return
    
#     try:
#         is_processing = True
        
#         # Play a quick acknowledgment sound
#         play_acknowledgment_sound()
        
#         # 1. Capture an image
#         logger.info("'Insight' detected - capturing image")
#         image_stream = capture_image()
#         if not image_stream:
#             logger.error("Failed to capture image, aborting process")
#             return
        
#         # 2. Send to OCR server
#         audio_data = send_to_ocr_server(image_stream)
#         if not audio_data:
#             logger.error("Failed to get audio response, aborting process")
#             return
        
#         # 3. Play the response
#         play_audio_data(audio_data)
        
#         logger.info("OCR process completed successfully")
#     except Exception as e:
#         logger.error(f"Error in processing pipeline: {e}")
#     finally:
#         is_processing = False
#         logger.info("Returning to 'insight' wake word detection...")

# def main():
#     """Main function to run the wakeword detection and OCR client"""
#     logger.info("Starting OCR client with 'insight' wake word detection...")
    
#     # Make sure all dependencies are available
#     if not setup_dependencies():
#         logger.error("Failed to set up required dependencies. Exiting.")
#         return
    
#     # Avoid importing at module level to allow setup_dependencies to install
#     import pvporcupine
#     from pvrecorder import PvRecorder
    
#     # Keep track of resources for proper cleanup
#     porcupine = None
#     recorder = None
    
#     # Handle graceful shutdown
#     def signal_handler(sig, frame):
#         logger.info("Received shutdown signal, cleaning up...")
#         if recorder:
#             recorder.delete()
#         if porcupine:
#             porcupine.delete()
#         logger.info("Shutdown complete")
#         exit(0)
    
#     signal.signal(signal.SIGINT, signal_handler)
#     signal.signal(signal.SIGTERM, signal_handler)
    
#     try:
#         # Initialize Porcupine for wakeword detection
#         porcupine = pvporcupine.create(
#             access_key=ACCESS_KEY,
#             keyword_paths=[KEYWORD_PATH],
#             sensitivities=[SENSITIVITY]
#         )
        
#         # Initialize recorder
#         recorder = PvRecorder(frame_length=porcupine.frame_length, device_index=-1)
#         recorder.start()
        
#         logger.info("Listening for 'insight' wake word...")
        
#         while True:
#             # Listen for wakeword
#             pcm = recorder.read()
#             result = porcupine.process(pcm)
            
#             if result >= 0:
#                 logger.info("'Insight' wake word detected!")
                
#                 # To avoid audio conflicts, stop recording while processing
#                 recorder.stop()
                
#                 # Process the OCR request in a separate thread to avoid blocking
#                 process_thread = threading.Thread(target=process_wakeword_detected)
#                 process_thread.start()
#                 process_thread.join()  # Wait for processing to complete
                
#                 # Resume recording for next wakeword
#                 recorder.start()
#                 logger.info("Listening for 'insight' wake word again...")
                
#     except KeyboardInterrupt:
#         logger.info("Stopping due to keyboard interrupt...")
#     except Exception as e:
#         logger.error(f"Unexpected error in main loop: {e}")
#     finally:
#         # Clean up resources
#         if recorder:
#             recorder.delete()
#         if porcupine:
#             porcupine.delete()
#         logger.info("Resources cleaned up")

# if __name__ == "__main__":
#     main()