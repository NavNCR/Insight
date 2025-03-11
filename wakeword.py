import pvporcupine
from pvrecorder import PvRecorder

# Static variables
ACCESS_KEY = "APy3GFzbJMwPAskDcUB4qJQKIxdc9W0u5ENvlLdhFKm/98Flrmaj5g=="
KEYWORD_PATH = "wakeword.ppn"

# Initialize Porcupine
porcupine = pvporcupine.create(
    access_key=ACCESS_KEY,
    keyword_paths=[KEYWORD_PATH],
    sensitivities=[0.5]
)

recorder = PvRecorder(frame_length=porcupine.frame_length, device_index=-1)
recorder.start()

print("Listening for wake word...")

try:
    while True:
        pcm = recorder.read()
        if porcupine.process(pcm) >= 0:
            print("Wakeword detected!")
except KeyboardInterrupt:
    print("Stopping...")
finally:
    recorder.delete()
    porcupine.delete()
