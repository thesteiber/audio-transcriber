import pyaudio
import wave
import numpy as np
import threading
import time
import os
import requests
import tempfile
from pydub import AudioSegment
import openai
import soundcard as sc
import queue
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from tkinter.filedialog import asksaveasfilename
import json
import datetime

class AudioTranscriber:
    def __init__(self, api_key, chunk_duration=10):
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=self.api_key)
        self.chunk_duration = chunk_duration  # Duration of each audio chunk to transcribe
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.chunk = 1024
        self.audio = pyaudio.PyAudio()
        self.mic_queue = queue.Queue()
        self.speaker_queue = queue.Queue()
        self.is_recording = False
        self.mic_frames = []
        self.speaker_frames = []
        self.mic_chunk_ready = threading.Event()
        self.speaker_chunk_ready = threading.Event()
        self.mic_buffer = []
        self.speaker_buffer = []
        self.buffer_lock = threading.Lock()
        self.mic_transcription_thread = None
        self.speaker_transcription_thread = None
        
    def stop_recording(self):
        """Stop the recording process"""
        self.is_recording = False
        
    def record_from_microphone(self, device_id=None):
        """Record audio from the microphone in streaming mode"""
        try:
            # Create a stream with the specified input device (or default if None)
            stream_args = {
                "format": self.format,
                "channels": self.channels,
                "rate": self.rate,
                "input": True,
                "frames_per_buffer": self.chunk
            }
            if device_id is not None:
                stream_args["input_device_index"] = device_id
                
            stream = self.audio.open(**stream_args)
            
            print(f"Started streaming from microphone {device_id if device_id is not None else '(default)'}")
            self.is_recording = True
            chunk_start_time = time.time()
            current_chunk = []
            
            while self.is_recording:
                # Read audio data
                data = stream.read(self.chunk, exception_on_overflow=False)
                current_chunk.append(data)
                
                # Check if we've reached the chunk duration
                elapsed = time.time() - chunk_start_time
                if elapsed >= self.chunk_duration:
                    # Send chunk for transcription
                    with self.buffer_lock:
                        self.mic_buffer.append(current_chunk)
                    self.mic_chunk_ready.set()
                    
                    # Reset for next chunk
                    current_chunk = []
                    chunk_start_time = time.time()
                    
            # Handle any remaining audio in the current chunk
            if current_chunk:
                with self.buffer_lock:
                    self.mic_buffer.append(current_chunk)
                self.mic_chunk_ready.set()
                
            print("Microphone recording stopped")
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            print(f"Error recording from microphone: {e}")
            import traceback
            traceback.print_exc()
            
    def record_system_audio(self, device_id=None):
        """Record system audio in streaming mode"""
        try:
            speakers = sc.all_speakers()
            
            if not speakers:
                print("No speakers found!")
                return
                
            # Use specified device or default to first available
            speaker_to_use = None
            if device_id is not None:
                for speaker in speakers:
                    if str(speaker.id) == device_id:
                        speaker_to_use = speaker
                        break
            
            if speaker_to_use is None:
                speaker_to_use = speakers[0]
                
            print(f"Started streaming from system audio: {speaker_to_use.name}")
            self.is_recording = True
            chunk_start_time = time.time()
            current_chunk = []
            
            # Calculate frames for each chunk
            chunk_frames = int(self.rate * self.chunk_duration)
            record_chunk_size = min(chunk_frames, self.rate)  # Record in 1-second increments
            
            with sc.get_microphone(include_loopback=True, id=str(speaker_to_use.id)) as mic:
                print(f"Opened loopback device: {mic}")
                frames_in_current_chunk = 0
                
                while self.is_recording:
                    # Record an increment (up to 1 second)
                    data = mic.record(record_chunk_size)
                    
                    # Convert to int16 and then to bytes
                    scaled = np.int16(data * 32767)
                    for i in range(0, len(scaled), self.chunk):
                        end_idx = min(i + self.chunk, len(scaled))
                        chunk_data = scaled[i:end_idx]
                        # Pad the last chunk if needed
                        if len(chunk_data) < self.chunk:
                            chunk_data = np.pad(chunk_data, ((0, self.chunk - len(chunk_data)), (0, 0)), 'constant')
                        current_chunk.append(chunk_data[:, 0].tobytes())  # Use first channel for mono
                    
                    frames_in_current_chunk += record_chunk_size
                    
                    # Check if we've reached the chunk duration
                    elapsed = time.time() - chunk_start_time
                    if elapsed >= self.chunk_duration or frames_in_current_chunk >= chunk_frames:
                        # Send chunk for transcription
                        with self.buffer_lock:
                            self.speaker_buffer.append(current_chunk)
                        self.speaker_chunk_ready.set()
                        
                        # Reset for next chunk
                        current_chunk = []
                        chunk_start_time = time.time()
                        frames_in_current_chunk = 0
                        
            # Handle any remaining audio in the current chunk
            if current_chunk:
                with self.buffer_lock:
                    self.speaker_buffer.append(current_chunk)
                self.speaker_chunk_ready.set()
                
            print("System audio recording stopped")
            
        except Exception as e:
            print(f"Error recording system audio: {e}")
            import traceback
            traceback.print_exc()
            
    def transcribe_mic_chunks(self, callback=None):
        """Continuously transcribe microphone audio chunks"""
        while self.is_recording or self.mic_buffer:
            # Wait for a chunk to be ready
            self.mic_chunk_ready.wait(timeout=1.0)
            
            # Get a chunk if available
            current_chunk = None
            with self.buffer_lock:
                if self.mic_buffer:
                    current_chunk = self.mic_buffer.pop(0)
                    if not self.mic_buffer:  # If no more chunks, clear the event
                        self.mic_chunk_ready.clear()
            
            if current_chunk:
                try:
                    # Save audio chunk to temp file
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                        temp_filename = temp_file.name
                        
                    self.save_audio(current_chunk, temp_filename)
                    
                    # Transcribe the audio
                    transcription = self.transcribe_audio(temp_filename)
                    
                    # Call the callback with the transcription result
                    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                    if callback and transcription:
                        callback("mic", f"[{timestamp}] {transcription}")
                        
                    # Clean up temp file
                    os.unlink(temp_filename)
                    
                except Exception as e:
                    print(f"Error transcribing mic chunk: {e}")
                    import traceback
                    traceback.print_exc()
            
    def transcribe_speaker_chunks(self, callback=None):
        """Continuously transcribe system audio chunks"""
        while self.is_recording or self.speaker_buffer:
            # Wait for a chunk to be ready
            self.speaker_chunk_ready.wait(timeout=1.0)
            
            # Get a chunk if available
            current_chunk = None
            with self.buffer_lock:
                if self.speaker_buffer:
                    current_chunk = self.speaker_buffer.pop(0)
                    if not self.speaker_buffer:  # If no more chunks, clear the event
                        self.speaker_chunk_ready.clear()
            
            if current_chunk:
                try:
                    # Save audio chunk to temp file
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                        temp_filename = temp_file.name
                        
                    self.save_audio(current_chunk, temp_filename)
                    
                    # Transcribe the audio
                    transcription = self.transcribe_audio(temp_filename)
                    
                    # Call the callback with the transcription result
                    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                    if callback and transcription:
                        callback("speaker", f"[{timestamp}] {transcription}")
                        
                    # Clean up temp file
                    os.unlink(temp_filename)
                    
                except Exception as e:
                    print(f"Error transcribing speaker chunk: {e}")
                    import traceback
                    traceback.print_exc()
        
    def save_audio(self, frames, filename):
        """Save recorded frames to a WAV file"""
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        
    def transcribe_audio(self, audio_file):
        """Transcribe audio using OpenAI's Whisper API"""
        try:
            print(f"Transcribing file: {audio_file} (size: {os.path.getsize(audio_file)} bytes)")
            with open(audio_file, "rb") as file:
                transcription = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=file
                )
            print(f"Transcription completed")
            return transcription.text
        except Exception as e:
            print(f"Transcription error: {e}")
            import traceback
            traceback.print_exc()
            return ""
            
    def start_streaming(self, mic_device_id=None, speaker_device_id=None, 
                        use_mic=True, use_speaker=True, callback=None):
        """Start streaming audio transcription"""
        self.is_recording = True
        
        # Start recording threads
        if use_mic:
            mic_thread = threading.Thread(
                target=self.record_from_microphone,
                args=(mic_device_id,)
            )
            mic_thread.daemon = True
            mic_thread.start()
            
            # Start transcription thread for microphone
            self.mic_transcription_thread = threading.Thread(
                target=self.transcribe_mic_chunks,
                args=(callback,)
            )
            self.mic_transcription_thread.daemon = True
            self.mic_transcription_thread.start()
            
        if use_speaker:
            speaker_thread = threading.Thread(
                target=self.record_system_audio,
                args=(speaker_device_id,)
            )
            speaker_thread.daemon = True
            speaker_thread.start()
            
            # Start transcription thread for speaker
            self.speaker_transcription_thread = threading.Thread(
                target=self.transcribe_speaker_chunks,
                args=(callback,)
            )
            self.speaker_transcription_thread.daemon = True
            self.speaker_transcription_thread.start()
            
    def stop_streaming(self):
        """Stop streaming audio transcription"""
        self.is_recording = False
        
        # Generate one last event to wake up transcription threads
        self.mic_chunk_ready.set()
        self.speaker_chunk_ready.set()
            
        # Wait for transcription threads to finish
        if self.mic_transcription_thread and self.mic_transcription_thread.is_alive():
            self.mic_transcription_thread.join(timeout=5.0)
            
        if self.speaker_transcription_thread and self.speaker_transcription_thread.is_alive():
            self.speaker_transcription_thread.join(timeout=5.0)
            
        print("Streaming stopped")
        self.audio.terminate()

class AudioTranscriberGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Streaming Audio Transcriber")
        self.root.geometry("800x600")
        self.root.minsize(600, 400)
        
        self.transcriber = None
        self.streaming = False
        
        # Get available audio devices
        self.refresh_devices()
        
        self.create_widgets()
        self.load_config()
        
    def refresh_devices(self):
        """Get available audio input and output devices"""
        self.audio = pyaudio.PyAudio()
        
        # Get microphone devices
        self.mic_devices = []
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            if device_info.get('maxInputChannels') > 0:  # Check if it's an input device
                device_name = device_info.get('name', f"Microphone {i}")
                self.mic_devices.append((i, device_name))
                
        # Get speaker devices using soundcard
        self.speaker_devices = []
        try:
            speakers = sc.all_speakers()
            for i, speaker in enumerate(speakers):
                self.speaker_devices.append((str(speaker.id), speaker.name))
        except Exception as e:
            print(f"Error getting speaker devices: {e}")
        
    def create_widgets(self):
        # Create a main frame with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10")
        settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # API Key
        ttk.Label(settings_frame, text="OpenAI API Key:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.api_key_var = tk.StringVar()
        self.api_key_entry = ttk.Entry(settings_frame, textvariable=self.api_key_var, width=40, show="*")
        self.api_key_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Show/Hide API Key
        self.show_api_key_var = tk.BooleanVar()
        self.show_api_key_check = ttk.Checkbutton(settings_frame, text="Show API Key", 
                                                  variable=self.show_api_key_var, 
                                                  command=self.toggle_api_key_visibility)
        self.show_api_key_check.grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        
        # Chunk Duration
        ttk.Label(settings_frame, text="Chunk Duration (seconds):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.duration_var = tk.StringVar(value="10")
        self.duration_entry = ttk.Spinbox(settings_frame, from_=5, to=30, textvariable=self.duration_var, width=5)
        self.duration_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Device selection frame
        device_frame = ttk.LabelFrame(main_frame, text="Audio Devices", padding="10")
        device_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Microphone selection
        ttk.Label(device_frame, text="Microphone:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.mic_var = tk.StringVar()
        self.mic_dropdown = ttk.Combobox(device_frame, textvariable=self.mic_var, width=40)
        self.mic_dropdown.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Populate microphone dropdown
        mic_options = [f"{name} (ID: {index})" for index, name in self.mic_devices]
        self.mic_dropdown['values'] = mic_options
        if mic_options:
            self.mic_dropdown.current(0)
            
        # Speaker selection
        ttk.Label(device_frame, text="System Audio:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.speaker_var = tk.StringVar()
        self.speaker_dropdown = ttk.Combobox(device_frame, textvariable=self.speaker_var, width=40)
        self.speaker_dropdown.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Populate speaker dropdown
        speaker_options = [f"{name} (ID: {id})" for id, name in self.speaker_devices]
        self.speaker_dropdown['values'] = speaker_options
        if speaker_options:
            self.speaker_dropdown.current(0)
            
        # Refresh devices button
        self.refresh_button = ttk.Button(device_frame, text="Refresh Devices", command=self.update_devices)
        self.refresh_button.grid(row=0, column=2, rowspan=2, padx=5, pady=5)
        
        # Record from Microphone
        self.record_mic_var = tk.BooleanVar(value=True)
        self.record_mic_check = ttk.Checkbutton(device_frame, text="Record from Microphone", 
                                               variable=self.record_mic_var)
        self.record_mic_check.grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        
        # Record from System Audio
        self.record_system_var = tk.BooleanVar(value=True)
        self.record_system_check = ttk.Checkbutton(device_frame, text="Record from System Audio", 
                                                  variable=self.record_system_var)
        self.record_system_check.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Controls frame
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Stream button
        self.stream_button = ttk.Button(controls_frame, text="Start Streaming", command=self.toggle_streaming)
        self.stream_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Clear button
        self.clear_button = ttk.Button(controls_frame, text="Clear Logs", command=self.clear_logs)
        self.clear_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Save button
        self.save_button = ttk.Button(controls_frame, text="Save Logs", command=self.save_logs)
        self.save_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(controls_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Transcription logs
        logs_frame = ttk.LabelFrame(main_frame, text="Transcription Logs", padding="10")
        logs_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Logs notebook
        self.logs_notebook = ttk.Notebook(logs_frame)
        self.logs_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Microphone tab
        self.mic_tab = ttk.Frame(self.logs_notebook)
        self.logs_notebook.add(self.mic_tab, text="Microphone")
        
        self.mic_text = scrolledtext.ScrolledText(self.mic_tab, wrap=tk.WORD)
        self.mic_text.pack(fill=tk.BOTH, expand=True)
        
        # System audio tab
        self.system_tab = ttk.Frame(self.logs_notebook)
        self.logs_notebook.add(self.system_tab, text="System Audio")
        
        self.system_text = scrolledtext.ScrolledText(self.system_tab, wrap=tk.WORD)
        self.system_text.pack(fill=tk.BOTH, expand=True)
        
        # Combined tab
        self.combined_tab = ttk.Frame(self.logs_notebook)
        self.logs_notebook.add(self.combined_tab, text="Combined")
        
        self.combined_text = scrolledtext.ScrolledText(self.combined_tab, wrap=tk.WORD)
        self.combined_text.pack(fill=tk.BOTH, expand=True)
        
    def toggle_api_key_visibility(self):
        if self.show_api_key_var.get():
            self.api_key_entry.config(show="")
        else:
            self.api_key_entry.config(show="*")
            
    def load_config(self):
        try:
            if os.path.exists("audio_transcriber_config.json"):
                with open("audio_transcriber_config.json", "r") as f:
                    config = json.load(f)
                    self.api_key_var.set(config.get("api_key", ""))
                    self.duration_var.set(str(config.get("chunk_duration", 10)))
                    
                    # Test API connection
                    api_key = config.get("api_key", "")
                    if api_key:
                        print("Testing API connection...")
                        try:
                            client = openai.OpenAI(api_key=api_key)
                            models = client.models.list()
                            print(f"API connection successful. Available models: {[m.id for m in models.data[:3]]}")
                        except Exception as e:
                            print(f"API connection test failed: {e}")
        except Exception as e:
            print(f"Error loading config: {e}")
            
    def save_config(self):
        try:
            config = {
                "api_key": self.api_key_var.get(),
                "chunk_duration": int(self.duration_var.get())
            }
            with open("audio_transcriber_config.json", "w") as f:
                json.dump(config, f)
        except Exception as e:
            print(f"Error saving config: {e}")
            
    def update_devices(self):
        """Refresh device lists and update dropdowns"""
        # Clear existing devices
        self.audio.terminate()
        
        # Refresh device lists
        self.refresh_devices()
        
        # Update microphone dropdown
        mic_options = [f"{name} (ID: {index})" for index, name in self.mic_devices]
        self.mic_dropdown['values'] = mic_options
        if mic_options:
            self.mic_dropdown.current(0)
            
        # Update speaker dropdown
        speaker_options = [f"{name} (ID: {id})" for id, name in self.speaker_devices]
        self.speaker_dropdown['values'] = speaker_options
        if speaker_options:
            self.speaker_dropdown.current(0)
            
        messagebox.showinfo("Devices Refreshed", f"Found {len(mic_options)} microphones and {len(speaker_options)} speakers")
        
    def toggle_streaming(self):
        if not self.streaming:
            # Start streaming
            api_key = self.api_key_var.get().strip()
            if not api_key:
                messagebox.showerror("Error", "Please enter your OpenAI API key")
                return
                
            try:
                chunk_duration = int(self.duration_var.get())
                if chunk_duration <= 0:
                    raise ValueError("Chunk duration must be positive")
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid chunk duration")
                return
                
            if not self.record_mic_var.get() and not self.record_system_var.get():
                messagebox.showerror("Error", "Please select at least one audio source")
                return
                
            self.save_config()
            self.streaming = True
            self.stream_button.config(text="Stop Streaming")
            self.status_var.set("Streaming...")
            
            # Get selected device IDs
            mic_id = None
            if self.mic_devices and self.record_mic_var.get():
                selected_mic = self.mic_dropdown.current()
                if selected_mic >= 0:
                    mic_id = self.mic_devices[selected_mic][0]
                    
            speaker_id = None
            if self.speaker_devices and self.record_system_var.get():
                selected_speaker = self.speaker_dropdown.current()
                if selected_speaker >= 0:
                    speaker_id = self.speaker_devices[selected_speaker][0]
            
            # Start the transcriber
            self.transcriber = AudioTranscriber(api_key, chunk_duration)
            self.transcriber.start_streaming(
                mic_device_id=mic_id,
                speaker_device_id=speaker_id,
                use_mic=self.record_mic_var.get(),
                use_speaker=self.record_system_var.get(),
                callback=self.update_transcription
            )
            
        else:
            # Stop streaming
            self.streaming = False
            self.stream_button.config(text="Start Streaming")
            self.status_var.set("Stopping streaming...")
            
            if self.transcriber:
                self.transcriber.stop_streaming()
                
            self.status_var.set("Ready")
            
    def update_transcription(self, source, text):
        """Update the transcription logs with new text"""
        # Add text to appropriate log
        if source == "mic":
            self.mic_text.insert(tk.END, text + "\n\n")
            self.mic_text.see(tk.END)
        elif source == "speaker":
            self.system_text.insert(tk.END, text + "\n\n")
            self.system_text.see(tk.END)
            
        # Add to combined log with source indicator
        if source == "mic":
            combined_text = f"[MIC] {text}\n\n"
        else:
            combined_text = f"[SPEAKER] {text}\n\n"
            
        self.combined_text.insert(tk.END, combined_text)
        self.combined_text.see(tk.END)
        
    def clear_logs(self):
        """Clear all transcription logs"""
        self.mic_text.delete(1.0, tk.END)
        self.system_text.delete(1.0, tk.END)
        self.combined_text.delete(1.0, tk.END)
        
    def save_logs(self):
        """Save transcription logs to a file"""
        try:
            filename = asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Save Transcription Logs"
            )
            
            if not filename:
                return
                
            with open(filename, "w", encoding="utf-8") as f:
                # Write a header with timestamp
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"=== Transcription Logs - {timestamp} ===\n\n")
                
                # Write combined logs
                f.write("=== Combined Logs ===\n\n")
                f.write(self.combined_text.get(1.0, tk.END))
                
                # Write individual logs
                f.write("\n\n=== Microphone Logs ===\n\n")
                f.write(self.mic_text.get(1.0, tk.END))
                
                f.write("\n\n=== System Audio Logs ===\n\n")
                f.write(self.system_text.get(1.0, tk.END))
                
            messagebox.showinfo("Success", f"Logs saved to {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error saving logs: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioTranscriberGUI(root)
    root.mainloop() 