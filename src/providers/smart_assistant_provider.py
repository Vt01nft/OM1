"""Smart Assistant Provider for OM1.

Bounty #367: OM1 + Smart Assistant + Wallet Payments

This provider enables OM1 to integrate with a smart assistant that supports
voice and text interactions for ordering products and processing payments
through connected crypto wallets.

Features:
- Voice recognition (Google Speech API)
- Text-to-Speech (pyttsx3)
- Natural language command processing
- Friendly conversational responses
- Integration with wallet providers for payments
"""

import logging
import random
from typing import Any, Callable, Dict, Optional

try:
    from providers.singleton import singleton
except ImportError:

    def singleton(cls):
        """Singleton decorator fallback."""
        return cls


# Optional voice dependencies
TTS_AVAILABLE = False
STT_AVAILABLE = False

try:
    import pyttsx3

    TTS_AVAILABLE = True
except ImportError:
    pass

try:
    import speech_recognition as sr

    STT_AVAILABLE = True
except ImportError:
    pass

logger = logging.getLogger(__name__)


@singleton
class SmartAssistantProvider:
    """Smart Assistant Provider for voice and text interactions.

    Provides natural language interface for OM1 to communicate with users,
    process orders, and trigger wallet payments.

    Example:
        assistant = SmartAssistantProvider(user_name="Alice")
        assistant.set_voice_mode(True)
        response = assistant.process_command("order a coffee")
    """

    # Greeting responses
    GREETINGS = [
        "Hello {name}! How can I help you today?",
        "Hey {name}! What would you like to do?",
        "Hi there! Ready to assist you!",
    ]

    # Thank you responses
    THANKS_RESPONSES = [
        "You're welcome! Anything else I can help with?",
        "My pleasure! Let me know if you need more help!",
        "Anytime! Happy to help!",
    ]

    # Goodbye responses
    GOODBYE_RESPONSES = [
        "Goodbye {name}! Have a great day!",
        "See you later {name}! Take care!",
        "Bye! Come back anytime!",
    ]

    # Order confirmation responses
    ORDER_CONFIRMATIONS = [
        "Payment confirmed! Your {item} is on its way! Enjoy!",
        "Success! One {item} coming right up!",
        "All done! Your {item} will be ready soon!",
    ]

    def __init__(
        self,
        user_name: str = "Friend",
        voice_enabled: bool = True,
        tts_rate: int = 160,
    ):
        """Initialize the Smart Assistant Provider.

        Args:
            user_name: Name to address the user.
            voice_enabled: Whether to enable voice features.
            tts_rate: Speech rate for text-to-speech (words per minute).
        """
        self.user_name = user_name
        self.voice_mode = False
        self.tts_rate = tts_rate

        # Initialize TTS
        self.tts_engine = None
        if voice_enabled and TTS_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty("rate", tts_rate)
                logger.info("TTS engine initialized")
            except Exception as e:
                logger.warning(f"TTS initialization failed: {e}")

        # Initialize STT
        self.recognizer = None
        self.microphone = None
        if voice_enabled and STT_AVAILABLE:
            try:
                self.recognizer = sr.Recognizer()
                self.microphone = sr.Microphone()
                with self.microphone as src:
                    self.recognizer.adjust_for_ambient_noise(src, duration=1)
                logger.info("Speech recognition initialized")
            except Exception as e:
                logger.warning(f"Speech recognition initialization failed: {e}")

        # Command handlers
        self._command_handlers: Dict[str, Callable] = {}

        # Order callback
        self._order_callback: Optional[Callable] = None
        self._payment_callback: Optional[Callable] = None

        logger.info(f"SmartAssistantProvider initialized for user: {user_name}")

    def set_voice_mode(self, enabled: bool) -> None:
        """Enable or disable voice mode."""
        self.voice_mode = enabled
        logger.info(f"Voice mode: {'enabled' if enabled else 'disabled'}")

    def set_user_name(self, name: str) -> None:
        """Set the user's name for personalized responses."""
        self.user_name = name

    def register_order_callback(self, callback: Callable) -> None:
        """Register callback for order processing."""
        self._order_callback = callback

    def register_payment_callback(self, callback: Callable) -> None:
        """Register callback for payment processing."""
        self._payment_callback = callback

    def register_command(self, keywords: list, handler: Callable) -> None:
        """Register a custom command handler."""
        for keyword in keywords:
            self._command_handlers[keyword] = handler

    # ─────────────────────────────────────────────────────────────
    # Speech Methods
    # ─────────────────────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """Speak text aloud using TTS.

        Args:
            text: Text to speak.
        """
        if self.tts_engine and self.voice_mode:
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                logger.error(f"TTS error: {e}")

    def speak_fresh(self, text: str) -> None:
        """Speak text using a fresh TTS engine (more reliable).

        Args:
            text: Text to speak.
        """
        if TTS_AVAILABLE and self.voice_mode:
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", self.tts_rate)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                logger.error(f"TTS error: {e}")

    def listen(self, timeout: int = 8) -> Optional[str]:
        """Listen for voice input.

        Args:
            timeout: Maximum seconds to wait for input.

        Returns
        -------
            Recognized text or None if failed.
        """
        if not self.recognizer or not self.microphone:
            return None

        try:
            with self.microphone as src:
                audio = self.recognizer.listen(
                    src, timeout=timeout, phrase_time_limit=15
                )
            text = self.recognizer.recognize_google(audio)
            logger.info(f"Recognized: {text}")
            return text.lower()
        except sr.WaitTimeoutError:
            logger.debug("Listen timeout")
            return None
        except sr.UnknownValueError:
            logger.debug("Could not understand audio")
            return None
        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
            return None

    # ─────────────────────────────────────────────────────────────
    # Output Methods
    # ─────────────────────────────────────────────────────────────

    def output(self, text: str, print_text: bool = True) -> str:
        """Output response - prints and speaks in voice mode.

        Args:
            text: Text to output.
            print_text: Whether to print the text.

        Returns
        -------
            The output text.
        """
        if print_text:
            print(f"   🤖 {text}")

        if self.voice_mode:
            self.speak_fresh(text)

        return text

    def get_input(self, prompt: str = "") -> Optional[str]:
        """Get user input - voice in voice mode, text otherwise.

        Args:
            prompt: Prompt to display/speak.

        Returns
        -------
            User input text.
        """
        if prompt:
            self.output(prompt)

        if self.voice_mode:
            result = self.listen()
            if result:
                return result
            # Fallback to text
            return input("   ⌨️ Type: ").strip().lower()
        else:
            return input("   👤 You: ").strip().lower()

    # ─────────────────────────────────────────────────────────────
    # Response Generators
    # ─────────────────────────────────────────────────────────────

    def greet(self) -> str:
        """Generate a greeting response."""
        response = random.choice(self.GREETINGS).format(name=self.user_name)
        return self.output(response)

    def thank(self) -> str:
        """Generate a thank you response."""
        response = random.choice(self.THANKS_RESPONSES)
        return self.output(response)

    def goodbye(self) -> str:
        """Generate a goodbye response."""
        response = random.choice(self.GOODBYE_RESPONSES).format(name=self.user_name)
        return self.output(response)

    def confirm_order(self, item: str) -> str:
        """Generate an order confirmation response."""
        response = random.choice(self.ORDER_CONFIRMATIONS).format(item=item)
        return self.output(response)

    # ─────────────────────────────────────────────────────────────
    # Command Processing
    # ─────────────────────────────────────────────────────────────

    def process_command(self, text: str) -> Dict[str, Any]:
        """Process a user command and return structured response.

        Args:
            text: User command text.

        Returns
        -------
            Dictionary with intent, response, action, and data.
        """
        t = text.lower().strip()
        result = {
            "intent": "unknown",
            "response": "",
            "action": None,
            "data": {},
        }

        # Greetings
        if any(t == w or t.startswith(w + " ") for w in ["hi", "hello", "hey"]):
            result["intent"] = "greeting"
            result["response"] = self.greet()
            return result

        # How are you
        if any(p in t for p in ["how are you", "what's up", "whats up"]):
            result["intent"] = "greeting"
            result["response"] = self.output(
                f"I'm great, thanks for asking! How can I help you {self.user_name}?"
            )
            return result

        # Thank you
        if any(w in t for w in ["thank", "thanks"]):
            result["intent"] = "thanks"
            result["response"] = self.thank()
            return result

        # Goodbye
        if any(w in t for w in ["bye", "goodbye", "see you"]):
            result["intent"] = "goodbye"
            result["response"] = self.goodbye()
            return result

        # Balance check
        if any(w in t for w in ["balance", "how much"]):
            result["intent"] = "balance"
            result["action"] = "check_balance"
            result["data"]["token"] = self._detect_token(t)
            return result

        # Show wallets
        if any(w in t for w in ["wallet", "address"]):
            result["intent"] = "wallets"
            result["action"] = "show_wallets"
            return result

        # Order
        if any(w in t for w in ["order", "buy", "get me", "i want"]):
            result["intent"] = "order"
            result["action"] = "start_order"
            result["data"]["product"] = self._detect_product(t)
            return result

        # Send tokens
        if "send" in t:
            result["intent"] = "send"
            result["action"] = "send_tokens"
            result["data"]["token"] = self._detect_token(t)
            return result

        # Contacts
        if "contact" in t:
            result["intent"] = "contacts"
            if "save" in t or "add" in t:
                result["action"] = "save_contact"
            elif "delete" in t:
                result["action"] = "delete_contact"
            else:
                result["action"] = "show_contacts"
            return result

        # History
        if any(w in t for w in ["history", "transaction", "recent"]):
            result["intent"] = "history"
            result["action"] = "show_history"
            return result

        # Help
        if "help" in t:
            result["intent"] = "help"
            result["response"] = self.output(
                "I can check balance, show wallets, order food, send tokens, "
                "manage contacts, and show history!"
            )
            return result

        # Check custom handlers
        for keyword, handler in self._command_handlers.items():
            if keyword in t:
                result["intent"] = "custom"
                result["action"] = handler
                return result

        # Unknown
        result["response"] = self.output(
            "Sorry, I didn't understand. Try: balance, order, wallets, or send."
        )
        return result

    def _detect_token(self, text: str) -> Optional[str]:
        """Detect cryptocurrency token from text."""
        t = text.lower()
        if any(w in t for w in ["solana", "sol"]):
            return "SOL"
        if any(w in t for w in ["ethereum", "eth"]):
            return "ETH"
        if "usdc" in t:
            return "USDC"
        if "usdt" in t:
            return "USDT"
        return None

    def _detect_product(self, text: str) -> Optional[str]:
        """Detect product from text."""
        products = ["coffee", "tea", "pizza", "burger"]
        for p in products:
            if p in text.lower():
                return p
        return None

    # ─────────────────────────────────────────────────────────────
    # Conversation Flow
    # ─────────────────────────────────────────────────────────────

    def ask_choice(
        self, prompt: str, options: list, max_attempts: int = 3
    ) -> Optional[str]:
        """Ask user to choose from options.

        Args:
            prompt: Question to ask.
            options: List of valid options.
            max_attempts: Maximum retry attempts.

        Returns
        -------
            Selected option or None.
        """
        self.output(prompt)

        for attempt in range(max_attempts):
            response = self.get_input("Your choice?")
            if not response:
                continue

            # Check for match
            for opt in options:
                if opt.lower() in response or response in opt.lower():
                    return opt

            if attempt < max_attempts - 1:
                self.output(f"Please choose from: {', '.join(options)}")

        return None

    def ask_confirmation(self, prompt: str) -> bool:
        """Ask for yes/no confirmation.

        Args:
            prompt: Confirmation question.

        Returns
        -------
            True if confirmed, False otherwise.
        """
        self.output(f"{prompt} Say yes or no.")
        response = self.get_input("Confirm?")

        if response and any(
            w in response for w in ["yes", "yeah", "sure", "ok", "yep"]
        ):
            return True
        return False

    # ─────────────────────────────────────────────────────────────
    # Status
    # ─────────────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Get provider status."""
        return {
            "user_name": self.user_name,
            "voice_mode": self.voice_mode,
            "tts_available": TTS_AVAILABLE and self.tts_engine is not None,
            "stt_available": STT_AVAILABLE and self.recognizer is not None,
        }

    def __repr__(self) -> str:
        """Return string representation."""
        return f"SmartAssistantProvider(user={self.user_name}, voice={self.voice_mode})"
