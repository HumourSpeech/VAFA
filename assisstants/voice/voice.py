import sys
from assisstants.logging.logger import logging
from assisstants.exception.exception import AssisstantException

import threading
import speech_recognition as sr
import logging
import sys
from typing import List

class speech_to_text:
    try:
        def __init__(self):
            self.listening = False
            self.listener_thread = None
            self._transcripts: List[str] = []  # Store transcripts
            self._lock = threading.Lock()  # Thread lock for thread-safe operations

        def listen_in_background(self):
            r = sr.Recognizer()
            with sr.Microphone() as mic:
                while self.listening:
                    try:
                        r.adjust_for_ambient_noise(mic, duration=0.5)
                        audio = r.listen(mic, timeout=2, phrase_time_limit=5)
                        text = r.recognize_google(audio)
                        print("You said:", text)

                        with self._lock:
                            self._transcripts.append(text)

                    except sr.WaitTimeoutError:
                        continue
                    except sr.UnknownValueError:
                        self._transcripts.append("[Unrecognized Speech]")
                    except sr.RequestError as e:
                        self._transcripts.append("[API Error]")
                        break

        def start_listening(self):
            logging.info("Start listening in background thread")
            if not self.listening:
                self.listening = True
                self.listener_thread = threading.Thread(target=self.listen_in_background)
                self.listener_thread.start()
            logging.info("Listening started...")

        def stop_listening(self):
            """Stop listening and wait for the background thread to finish"""
            logging.info("Stop listening")
            self.listening = False
            
            # Wait for thread to finish (similar to original code)
            if self.listener_thread and self.listener_thread.is_alive():
                self.listener_thread.join(timeout=2.0)  # Wait up to 2 seconds
                if self.listener_thread.is_alive():
                    logging.warning("Listener thread did not exit within timeout.")
                else:
                    logging.info("Listener thread stopped successfully.")
            else:
                logging.info("No active listener thread to stop.")
            
            logging.info("Listening stopped.")

        def get_transcripts(self) -> List[str]:
            """Return a copy of collected transcripts"""
            with self._lock:
                return list(self._transcripts)  # Return a copy to avoid external modification

        def clear_transcripts(self):
            """Clear all stored transcripts"""
            with self._lock:
                self._transcripts.clear()
            logging.info("Transcripts cleared")

    except Exception as e:
        raise AssisstantException(e, sys)