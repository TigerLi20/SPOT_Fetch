import sounddevice as sd
import numpy as np
import whisper
from scipy.io.wavfile import write
import time
from scipy.signal import resample

# Set up Whisper model
model = whisper.load_model("base.en")  # Use "tiny", "base", "small", "medium", or "large"

default_device_index = sd.default.device[0]  # Get default input device index
default_input_device = sd.query_devices(default_device_index)


# Audio configuration
INPUT_SAMPLE_RATE = int(default_input_device['default_samplerate'])  # Default microphone sample rate
WHISPER_SAMPLE_RATE = 16000  # Whisper expects 16 kHz
CHUNK_DURATION = 5  # seconds
CHUNK_SIZE = INPUT_SAMPLE_RATE * CHUNK_DURATION
SILENCE_THRESHOLD = 0.1  # Amplitude below which it's considered silence
SILENCE_TIMEOUT = 20  # Timeout for silence detection (seconds)
DEFAULT_TRANSCRIPTION = "Find index of all objects that can manipulated by one hand."


# Buffer to hold audio chunks
audio_buffer = []
voice_detected = False  # Flag to track if any voice was detected

def callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}")
    audio_data = indata[:, 0]  # Extract mono audio
    audio_buffer.extend(audio_data)
def clear_transcription(filename="./.tmp/.transcriptions.txt"):
    with open(filename, "w") as f:  # Open file in append mode
        f.write("")
def save_transcription(transcription, filename="./.tmp/.transcriptions.txt"):
    with open(filename, "a") as f:  # Open file in append mode
        f.write(transcription + "\n")

def is_silent(audio_data):
    """Check if the audio data contains mostly silence."""
    max_amplitude = np.max(np.abs(audio_data))
    print(f"Max Amplitude: {max_amplitude}")  # Log max amplitude for debugging
    return max_amplitude < SILENCE_THRESHOLD

def resample_audio(audio, input_rate, target_rate):
    """Resample audio data to the target sample rate."""
    num_samples = int(len(audio) * target_rate / input_rate)
    return resample(audio, num_samples)


def check_microphone():
    """Check if a valid input device (microphone) exists."""
    devices = sd.query_devices()
    input_devices = [d for d in devices if d['max_input_channels'] > 0]
    
    if len(input_devices) == 0:
        print("⚠️ No microphone detected. Transcription cannot start.")
        return False
    return True

def transcribe_live():
    if not check_microphone():
        print("❌ Skipping transcription due to missing microphone.")
        return
    
    global voice_detected
    clear_transcription()
    try:
        print("Starting live transcription...")
        with sd.InputStream(samplerate=INPUT_SAMPLE_RATE, channels=1, callback=callback, dtype="float32"):
            last_voice_time = time.time()

            while True:
                # Check if audio buffer has enough data
                if len(audio_buffer) >= CHUNK_SIZE:
                    audio_chunk = np.array(audio_buffer[:CHUNK_SIZE])
                    del audio_buffer[:CHUNK_SIZE]

                    # Resample audio to 16 kHz
                    audio_chunk_resampled = resample_audio(audio_chunk, INPUT_SAMPLE_RATE, WHISPER_SAMPLE_RATE)

                    if is_silent(audio_chunk_resampled) and not voice_detected:
                        print("Silence detected.")
                        # Check if silence timeout has elapsed
                        if not voice_detected and time.time() - last_voice_time > SILENCE_TIMEOUT:
                            print("No voice input detected. Saving default transcription.")
                            save_transcription(DEFAULT_TRANSCRIPTION)
                            break
                    else:
                        voice_detected = True  # Mark that voice input was detected

                        # Save chunk to temporary WAV file
                        temp_wav = "./.tmp/.temp_audio.wav"
                        scaled_audio = (audio_chunk_resampled * 32767).astype(np.int16)  # Properly scale and convert
                        write(temp_wav, WHISPER_SAMPLE_RATE, scaled_audio)
                        if not is_silent(audio_chunk_resampled):
                            # Transcribe using Whisper
                            last_voice_time = time.time()
                            result = model.transcribe(temp_wav)
                            transcription = result['text']
                            print(len(transcription))
                            print(f"Transcription: {transcription}")
                            # Save transcription to a file
                            save_transcription(transcription)
                        
                        if  time.time() - last_voice_time > SILENCE_TIMEOUT:
                            break
                        
    except KeyboardInterrupt:
        print("\nLive transcription stopped.")
    except Exception as e:
        print(f"Error: {e}")

# Start live transcription
if __name__ == "__main__":
    transcribe_live()