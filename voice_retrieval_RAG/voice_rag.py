import os
import time
import wave
import pyaudio
import json
from openai import OpenAI
from dotenv import load_dotenv
from retrieval import retrieve_and_rerank

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env file.")

client = OpenAI(api_key=OPENAI_API_KEY)

# Audio Configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000  # OpenAI TTS uses 24kHz by default for some formats, but we record at standard
RECORD_RATE = 44100
RECORD_SECONDS = 5 # Default duration if not using manual stop
WAVE_OUTPUT_FILENAME = "input.wav"

def record_audio_manual(filename=WAVE_OUTPUT_FILENAME):
    """
    Records audio until the user presses Enter.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RECORD_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("\n🔴 Recording... Press Enter to stop.")
    frames = []

    try:
        while True:
            # We need to read in a non-blocking way or use a separate thread to check for input
            # For simplicity in this script, we'll just record for a fixed duration or use a loop
            # that checks for a keypress if we were using a library like `keyboard` (root required).
            # Since we are in a terminal, standard input blocking makes this hard without threads.
            # Let's use a simple "Record for X seconds" or "Ctrl+C to stop" approach for MVP.
            # OR: Use a separate thread for recording.
            
            data = stream.read(CHUNK)
            frames.append(data)
            
            # Check for stop condition? 
            # For MVP, let's just record for 5 seconds or handle KeyboardInterrupt
            pass
    except KeyboardInterrupt:
        print("\n⏹️ Recording stopped.")
    
    # Wait, the above loop is infinite and blocks.
    # Let's implement a simpler "Record for 5 seconds" for now to test.
    pass

def record_audio_fixed(filename=WAVE_OUTPUT_FILENAME, duration=5):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RECORD_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print(f"\n🔴 Recording for {duration} seconds...")
    frames = []

    for i in range(0, int(RECORD_RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("⏹️ Recording finished.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RECORD_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return filename

def transcribe_audio(filename):
    print("📝 Transcribing...")
    with open(filename, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file
        )
    return transcript.text

def generate_response(query, context_chunks):
    print("🧠 Generating response...")
    
    context_text = "\n\n".join([chunk["text"] for chunk in context_chunks])
    
    system_prompt = """You are a helpful assistant for the KMC RAG system. 
    Answer the user's question based ONLY on the provided context. 
    If the answer is not in the context, say you don't know.
    Keep your answer concise and suitable for voice output (avoid long lists or complex formatting)."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o", # Or gpt-3.5-turbo
        messages=messages
    )
    return response.choices[0].message.content

def speak_text(text):
    print("🔊 Speaking...")
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
        response_format="pcm" # Raw PCM data
    )
    
    # Play audio using PyAudio
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, 
                    channels=1, 
                    rate=24000, 
                    output=True)
    
    # Stream the response
    for chunk in response.iter_bytes(chunk_size=1024):
        stream.write(chunk)
        
    stream.stop_stream()
    stream.close()
    p.terminate()

def main():
    while True:
        input("\nPress Enter to start recording (or Ctrl+C to exit)...")
        
        # 1. Record
        audio_file = record_audio_fixed(duration=5)
        
        # 2. Transcribe
        text = transcribe_audio(audio_file)
        print(f"🗣️ You said: {text}")
        
        if not text.strip():
            print("No speech detected.")
            continue

        # 3. Retrieve
        context = retrieve_and_rerank(text)
        
        if not context:
            print("No relevant context found.")
            response_text = "I couldn't find any information about that in the database."
        else:
            # 4. Generate
            response_text = generate_response(text, context)
            
        print(f"🤖 AI: {response_text}")
        
        # 5. Speak
        speak_text(response_text)

if __name__ == "__main__":
    main()
