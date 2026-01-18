"""
Voice Assistant Provider for OM1.

This module provides voice recognition and text-to-speech functionality
for OM1 agents, enabling voice-triggered payments and commands.

Part of Bounty #367: OM1 + Smart Assistant + Wallet Payments
"""

import logging
import re
from typing import Any, Callable, Dict, Optional

try:
    import pyttsx3
    import speech_recognition as sr
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    logging.warning("Voice libraries not available")

from .singleton import singleton


@singleton
class VoiceAssistantProvider:
    """
    Voice assistant for OM1 smart wallet integration.
    
    Enables:
    - Voice command recognition
    - Natural language payment parsing
    - Text-to-speech confirmations
    """

    # Known command patterns
    BALANCE_KEYWORDS = ['balance', 'how much', 'wallet', 'funds']
    PAYMENT_KEYWORDS = ['send', 'pay', 'transfer']
    HELP_KEYWORDS = ['help', 'commands', 'what can you do']
    QUIT_KEYWORDS = ['quit', 'exit', 'bye', 'goodbye', 'stop']
    HISTORY_KEYWORDS = ['history', 'transactions', 'recent']
    CONFIRM_KEYWORDS = ['yes', 'confirm', 'do it', 'proceed', 'ok', 'okay', 'yep', 'sure']
    CANCEL_KEYWORDS = ['no', 'cancel', 'nevermind', 'abort', 'nope']
    ADD_CONTACT_KEYWORDS = ['add contact', 'new contact', 'save contact', 'add recipient']
    LIST_CONTACTS_KEYWORDS = ['list contacts', 'show contacts', 'contacts', 'address book']

    def __init__(self, voice_rate: int = 150):
        """Initialize voice assistant."""
        self.voice_rate = voice_rate
        self._running = False
        self._command_handlers: Dict[str, Callable] = {}
        self._pending_confirmation: Optional[Dict] = None

        if VOICE_AVAILABLE:
            try:
                self.recognizer = sr.Recognizer()
                self.microphone = sr.Microphone()
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', voice_rate)
                
                voices = self.tts_engine.getProperty('voices')
                if voices:
                    self.tts_engine.setProperty('voice', voices[0].id)
                    
                logging.info("Voice assistant initialized with TTS")
            except Exception as e:
                logging.warning(f"TTS init failed: {e}")
                self.tts_engine = None
        else:
            self.recognizer = None
            self.microphone = None
            self.tts_engine = None

    def start(self) -> None:
        """Start the provider."""
        self._running = True
        logging.info("VoiceAssistantProvider started")

    def stop(self) -> None:
        """Stop the provider."""
        self._running = False
        logging.info("VoiceAssistantProvider stopped")

    def speak(self, text: str) -> None:
        """Speak text aloud using TTS."""
        print(f"🔊 Assistant: {text}")
        
        if self.tts_engine:
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                logging.error(f"TTS error: {e}")

    def listen(self, timeout: int = 5) -> Optional[str]:
        """Listen for voice input."""
        if not VOICE_AVAILABLE or not self.recognizer:
            return None

        try:
            with self.microphone as source:
                print("🎤 Listening...")
                audio = self.recognizer.listen(source, timeout=timeout)

            text = self.recognizer.recognize_google(audio)
            print(f"🎤 Heard: {text}")
            return text

        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            return None
        except Exception as e:
            logging.error(f"Listen error: {e}")
            return None

    def parse_command(self, text: str) -> Dict[str, Any]:
        """
        Parse natural language command.
        Preserves original case for wallet addresses.
        """
        if not text:
            return {'type': 'empty'}

        original_text = text.strip()
        text_lower = original_text.lower()

        # Check for confirmation/cancel (when pending)
        if self._pending_confirmation:
            if any(kw in text_lower for kw in self.CONFIRM_KEYWORDS):
                return {'type': 'confirm', 'raw_text': original_text}
            if any(kw in text_lower for kw in self.CANCEL_KEYWORDS):
                return {'type': 'cancel', 'raw_text': original_text}

        # Check for quit
        if any(kw in text_lower for kw in self.QUIT_KEYWORDS):
            return {'type': 'quit', 'raw_text': original_text}

        # Check for help
        if any(kw in text_lower for kw in self.HELP_KEYWORDS):
            return {'type': 'help', 'raw_text': original_text}

        # Check for add contact
        if any(kw in text_lower for kw in self.ADD_CONTACT_KEYWORDS):
            return self._parse_add_contact(original_text)

        # Check for list contacts
        if any(kw in text_lower for kw in self.LIST_CONTACTS_KEYWORDS):
            return {'type': 'list_contacts', 'raw_text': original_text}

        # Check for history
        if any(kw in text_lower for kw in self.HISTORY_KEYWORDS):
            return {'type': 'history', 'raw_text': original_text}

        # Check for balance
        if any(kw in text_lower for kw in self.BALANCE_KEYWORDS):
            return {'type': 'balance', 'raw_text': original_text}

        # Check for payment
        if any(kw in text_lower for kw in self.PAYMENT_KEYWORDS):
            return self._parse_payment(original_text)

        return {'type': 'unknown', 'raw_text': original_text}

    def _parse_payment(self, text: str) -> Dict[str, Any]:
        """
        Parse payment command preserving address case.
        
        Handles formats like:
        - "send 0.1 SOL to alice"
        - "pay 0.5 to 2nYLs5pEuN971n6MWJ1NLF5KRR2YrqWZpguoocavEsdf"
        - "transfer 1 sol alice"
        """
        text_lower = text.lower()
        
        # Extract amount (required)
        amount_match = re.search(r'(\d+(?:\.\d+)?)', text)
        if not amount_match:
            return {
                'type': 'payment',
                'error': 'No amount specified',
                'raw_text': text
            }
        
        amount = float(amount_match.group(1))
        
        # Find recipient - preserve original case!
        words = text.split()
        recipient = None
        
        # Method 1: Look for "to [recipient]"
        for i, word in enumerate(words):
            if word.lower() == 'to' and i + 1 < len(words):
                recipient = words[i + 1]
                break
        
        # Method 2: Last word that's not a keyword or number
        if not recipient:
            skip_words = {'send', 'pay', 'transfer', 'sol', 'solana', 'to'}
            for word in reversed(words):
                if word.lower() not in skip_words and not re.match(r'^\d+\.?\d*$', word):
                    recipient = word
                    break

        return {
            'type': 'payment',
            'amount': amount,
            'recipient': recipient,
            'raw_text': text
        }

    def _parse_add_contact(self, text: str) -> Dict[str, Any]:
        """
        Parse add contact command.
        
        Formats:
        - "add contact alice 2nYLs5pEuN971n6MWJ1NLF5KRR2YrqWZpguoocavEsdf"
        - "save contact bob address123..."
        """
        words = text.split()
        
        # Find name and address
        name = None
        address = None
        
        # Skip "add contact" keywords and find remaining words
        skip_words = {'add', 'contact', 'new', 'save', 'recipient'}
        remaining = [w for w in words if w.lower() not in skip_words]
        
        if len(remaining) >= 2:
            # First remaining word is name, second is address
            name = remaining[0].lower()
            address = remaining[1]  # Preserve case!
        elif len(remaining) == 1:
            # Could be just name or just address
            if len(remaining[0]) >= 32:
                address = remaining[0]
            else:
                name = remaining[0].lower()

        return {
            'type': 'add_contact',
            'name': name,
            'address': address,
            'raw_text': text
        }

    def set_pending_confirmation(self, data: Dict[str, Any]) -> None:
        """Set a pending confirmation."""
        self._pending_confirmation = data

    def get_pending_confirmation(self) -> Optional[Dict[str, Any]]:
        """Get pending confirmation data."""
        return self._pending_confirmation

    def clear_pending_confirmation(self) -> None:
        """Clear pending confirmation."""
        self._pending_confirmation = None

    def has_pending_confirmation(self) -> bool:
        """Check if there's a pending confirmation."""
        return self._pending_confirmation is not None

    def get_help_text(self) -> str:
        """Get help message."""
        return (
            "Available commands: "
            "Say 'check balance' to see your wallet. "
            "Say 'send' followed by amount and recipient, like 'send 0.1 SOL to alice'. "
            "Say 'add contact' followed by name and address to save a contact. "
            "Say 'contacts' to list saved contacts. "
            "Say 'history' to see recent transactions. "
            "Say 'quit' to exit."
        )