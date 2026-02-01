"""
Audio Service for attendance announcements
Uses pyttsx3 for offline text-to-speech
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import io
import base64

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    print("⚠️ pyttsx3 not installed. Audio announcements will be disabled.")

from config import ANNOUNCEMENT_LANGUAGE, ANNOUNCEMENT_RATE


class AudioService:
    """
    Text-to-speech service for attendance announcements.
    Generates audio that can be sent to phone speakers via WebRTC.
    """
    
    def __init__(self):
        self.engine = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._initialized = False
        
    def initialize(self) -> bool:
        """Initialize the TTS engine"""
        if not PYTTSX3_AVAILABLE:
            return False
            
        if self._initialized:
            return True
        
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', ANNOUNCEMENT_RATE)
            
            # Try to set voice
            voices = self.engine.getProperty('voices')
            if voices:
                # Try to find an English voice
                for voice in voices:
                    if 'english' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
            
            self._initialized = True
            print("✅ Audio service initialized")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize audio service: {e}")
            return False
    
    def speak(self, text: str):
        """Speak text synchronously (blocks until finished)"""
        if not self._initialized:
            if not self.initialize():
                print(f"[Audio] {text}")  # Fallback to console
                return
        
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Error speaking: {e}")
    
    async def speak_async(self, text: str):
        """Speak text asynchronously"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self.speak, text)
    
    def announce_attendance(self, name: str, status: str = "present"):
        """Generate attendance announcement"""
        if status == "present":
            text = f"{name}, marked present. Welcome!"
        elif status == "late":
            text = f"{name}, marked late."
        else:
            text = f"{name}, attendance recorded."
        
        self.speak(text)
    
    async def announce_attendance_async(self, name: str, status: str = "present"):
        """Generate attendance announcement asynchronously"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor, 
            self.announce_attendance, 
            name, 
            status
        )
    
    def close(self):
        """Clean up resources"""
        self._executor.shutdown(wait=False)
        if self.engine:
            try:
                self.engine.stop()
            except:
                pass
        self._initialized = False


# Global audio service instance
audio_service = AudioService()
