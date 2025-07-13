INSIGHT – OCR-Based Smart Glasses for the Visually Impaired 

This project is a wearable assistive technology designed to help visually impaired individuals access printed text using a smart glass system. The device captures images of the environment, 
extracts text using OCR, and reads it out loud through a speaker or headset using text-to-speech technology.

Project Objective
To build a real-time OCR-based smart glasses system using Raspberry Pi Zero 2W that captures printed text and reads it aloud, enhancing independence and mobility for visually impaired users.

Key Features
- Image capture via Pi Camera
- Optical Character Recognition (OCR) using Pytesseract
- Text-to-speech using Google Text-to-Speech (gTTS)
- Wake-word detection ("Hey Insight") via Picovoice Porcupine
- Cloud-based Flask server integration for OCR processing
- Hands-free audio output through USB headphones

 Technologies Used
- **Python** (main logic and integration)
- **Flask** (cloud-based processing server)
- **Pytesseract** (OCR)
- **gTTS** (speech synthesis)
- **Porcupine** (wake word detection)
- **pvrecorder** (audio input)
- **Raspberry Pi Zero 2 W**
- **Camera Module 3**
- **USB Audio Card + Headset**

System Architecture
1. Wake-word "Hey Insight" detected using Porcupine
2. Pi Camera captures image → sent to Flask server
3. OCR (Pytesseract) extracts text
4. gTTS converts text to MP3
5. Audio is sent back and played through headphones

Project Structure (Cloud Server)
Insight/
├── audio_files/
├── uploads/
├── client.py
├── wakeword.py
├── image_detection_integration.py
├── requirements.txt
└── flask_cloud/   

This project was inspired by and built upon concepts from various IEEE and academic papers on OCR, embedded vision systems, and smart assistive devices.

 Team Members
- Anjali Vinod 
- Navaneetha C R   
- Riya Mary Jose  
- Shivani R 

Institution
**Model Engineering College, Thrikkakara**  
APJ Abdul Kalam Technological University  
Electronics and Communication Engineering  
March 2025

