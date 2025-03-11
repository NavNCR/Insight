from flask import Flask, request, send_file, jsonify
import os
import tempfile
from PIL import Image
import pytesseract
from gtts import gTTS
import uuid
import io

# Set Tesseract executable path for Windows
# Comment this out if running on Raspberry Pi
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
AUDIO_FOLDER = 'audio_files'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

@app.route('/ocr', methods=['POST'])
def ocr_endpoint():
    # Check if image file is included in the request
    if 'image' not in request.files:
        return create_error_audio("No image file provided")
    
    file = request.files['image']
    
    # Check if file is empty
    if file.filename == '':
        return create_error_audio("Empty file provided")
    
    # Check if file is a JPEG
    if not file.filename.lower().endswith(('.jpg', '.jpeg')):
        return create_error_audio("Only JPEG images are accepted")
    
    try:
        # Save the uploaded image
        temp_image_path = os.path.join(UPLOAD_FOLDER, str(uuid.uuid4()) + '.jpg')
        file.save(temp_image_path)
        
        # Perform OCR on the image
        image = Image.open(temp_image_path)
        text = pytesseract.image_to_string(image)
        
        # Clean up the image file
        os.remove(temp_image_path)
        
        if not text.strip():
            return create_error_audio("No text detected in the image")
        
        # Convert text to speech
        audio_filename = str(uuid.uuid4()) + '.mp3'
        audio_path = os.path.join(AUDIO_FOLDER, audio_filename)
        tts = gTTS(text=text, lang='en')
        tts.save(audio_path)
        
        # Return the audio file
        return send_file(audio_path, mimetype='audio/mp3', as_attachment=True, 
                         download_name='ocr_result.mp3')
    
    except Exception as e:
        return create_error_audio(f"An error occurred: {str(e)}")

def create_error_audio(error_message):
    """Create an audio file from an error message"""
    print(f"Error: {error_message}")
    
    # Generate audio from error message
    audio_filename = str(uuid.uuid4()) + '.mp3'
    audio_path = os.path.join(AUDIO_FOLDER, audio_filename)
    
    tts = gTTS(text=error_message, lang='en')
    tts.save(audio_path)
    
    # Return the audio file
    return send_file(audio_path, mimetype='audio/mp3', as_attachment=True, 
                     download_name='error_message.mp3')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'OCR-TTS API'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)