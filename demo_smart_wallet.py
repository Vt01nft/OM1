"""
OM1 Smart Wallet + Voice Assistant + Store Demo

Bounty #367: OM1 + Smart Assistant + Wallet Payments

Full workflow: Order → Assistant → Payment → Confirmation
"""

import sys
import os
import re
import json
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional

sys.path.insert(0, 'src')

# Setup mock singleton
import types
mock = types.ModuleType('providers.singleton')
def singleton(cls): return cls
mock.singleton = singleton
sys.modules['providers.singleton'] = mock

# Load providers
def load_provider(filepath):
    with open(filepath, 'r') as f:
        code = f.read().replace('from .singleton import singleton', 'from providers.singleton import singleton')
        exec(code, globals())

load_provider('src/providers/solana_wallet_provider.py')
load_provider('src/providers/store_assistant_provider.py')

# Try to load Ethereum provider
try:
    load_provider('src/providers/ethereum_wallet_provider.py')
    ETH_AVAILABLE = True
except:
    ETH_AVAILABLE = False
    print("⚠️  Ethereum provider not available")

# Voice libraries
try:
    import speech_recognition as sr
    import pyttsx3
    VOICE_AVAILABLE = True
except:
    VOICE_AVAILABLE = False

# File paths
HISTORY_FILE = 'wallet_history.json'
CONTACTS_FILE = 'wallet_contacts.json'
ORDERS_FILE = 'order_history.json'


class PriceService:
    """Fetch live crypto prices."""
    
    def __init__(self):
        self.sol_price = 140.0
        self.eth_price = 3500.0
        self._fetch_prices()

    def _fetch_prices(self):
        try:
            url = "https://api.coingecko.com/api/v3/simple/price?ids=solana,ethereum&vs_currencies=usd"
            req = urllib.request.Request(url, headers={'User-Agent': 'OM1/1.0'})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                self.sol_price = data.get('solana', {}).get('usd', 140.0)
                self.eth_price = data.get('ethereum', {}).get('usd', 3500.0)
        except:
            pass

    def get_sol_price(self): return self.sol_price
    def get_eth_price(self): return self.eth_price
    def sol_to_usd(self, amount): return amount * self.sol_price
    def eth_to_usd(self, amount): return amount * self.eth_price
    def usd_to_sol(self, usd): return usd / self.sol_price
    def usd_to_eth(self, usd): return usd / self.eth_price


class VoiceInterface:
    """Voice input/output."""
    
    def __init__(self):
        self.recognizer = None
        self.microphone = None
        self.tts = None
        
        if VOICE_AVAILABLE:
            try:
                self.recognizer = sr.Recognizer()
                self.microphone = sr.Microphone()
                self.tts = pyttsx3.init()
                self.tts.setProperty('rate', 150)
                with self.microphone as src:
                    self.recognizer.adjust_for_ambient_noise(src, duration=0.5)
            except:
                pass

    def speak(self, text):
        print(f"🔊 Assistant: {text}")
        if self.tts:
            try:
                self.tts.say(text)
                self.tts.runAndWait()
            except:
                pass

    def listen(self, timeout=6):
        if not self.recognizer or not self.microphone:
            return None
        try:
            print("🎤 Listening...")
            with self.microphone as src:
                audio = self.recognizer.listen(src, timeout=timeout, phrase_time_limit=10)
            text = self.recognizer.recognize_google(audio)
            print(f"🎤 You said: {text}")
            return text
        except:
            return None

    def can_use_voice(self):
        return self.recognizer is not None


class Storage:
    """Persistent storage."""
    
    @staticmethod
    def load(filepath, default=None):
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    return json.load(f)
            except:
                pass
        return default if default else {}

    @staticmethod
    def save(filepath, data):
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        except:
            pass


class SmartWalletAssistant:
    """
    Full OM1 Smart Wallet + Store Assistant Demo.
    
    Features:
    - Voice/text commands
    - Product ordering
    - Multi-chain payments (SOL, USDC, ETH, USDT)
    - Transaction history
    - Contact management
    """

    STATES = {
        'NORMAL': 'normal',
        'PAYMENT_CONFIRM': 'payment_confirm',
        'ORDER_CONFIRM': 'order_confirm',
        'ORDER_PAYMENT': 'order_payment',
        'CONTACT_ADDRESS': 'contact_address',
        'CONTACT_NAME': 'contact_name',
    }

    DEFAULT_CONTACTS = {
        'alice': 'Hxro1u6mfWC4ZJJdxCgkEfGdJoc3nnsAw4Jebast1SX8',
        'bob': '5BZWY6XWPxuWFxs2jagkmUkCoBWmJ6c4YEArr83hYBWk',
    }

    def __init__(self, sol_private_key=None, eth_private_key=None):
        """Initialize everything."""
        print("\n🚀 Initializing OM1 Smart Wallet + Store Assistant...")
        
        # Initialize providers
        self.sol_wallet = SolanaWalletProvider(network='devnet', private_key=sol_private_key)
        self.store = StoreAssistantProvider()
        self.price = PriceService()
        self.voice = VoiceInterface()
        
        # ETH wallet (optional)
        self.eth_wallet = None
        if ETH_AVAILABLE and eth_private_key:
            try:
                self.eth_wallet = EthereumWalletProvider(network='sepolia', private_key=eth_private_key)
            except:
                pass

        # State
        self.state = self.STATES['NORMAL']
        self.pending = {}
        self.voice_mode = False

        # Load data
        self.contacts = {**self.DEFAULT_CONTACTS, **Storage.load(CONTACTS_FILE, {})}
        self.tx_history = Storage.load(HISTORY_FILE, [])
        self.order_history = Storage.load(ORDERS_FILE, [])

    def run(self):
        """Run the assistant."""
        self._print_banner()
        self.voice.speak("Smart wallet ready. You can check balance, send payments, or order products.")

        while True:
            try:
                text = self._get_input()
                if not text:
                    continue

                # Mode toggles
                if text.lower() in ['voice', 'voice mode']:
                    self._toggle_voice(True)
                    continue
                if text.lower() in ['text', 'text mode']:
                    self._toggle_voice(False)
                    continue

                result = self._process(text)
                if result.get('action') == 'quit':
                    self.voice.speak("Goodbye!")
                    break

            except KeyboardInterrupt:
                print("\n")
                self.voice.speak("Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                self.state = self.STATES['NORMAL']

    def _get_input(self):
        if self.voice_mode and self.voice.can_use_voice():
            result = self.voice.listen()
            if result:
                return result
            print("(No speech detected)")
        return input("🎤 You: ").strip()

    def _toggle_voice(self, enable):
        if enable and self.voice.can_use_voice():
            self.voice_mode = True
            self.voice.speak("Voice mode enabled.")
            print("🎤 VOICE MODE ON")
        else:
            self.voice_mode = False
            self.voice.speak("Text mode enabled.")
            print("⌨️  TEXT MODE ON")

    def _print_banner(self):
        sol_bal = self.sol_wallet.get_balance()
        sol_usd = self.price.sol_to_usd(sol_bal)
        
        print("\n" + "=" * 70)
        print("   OM1 SMART WALLET + STORE ASSISTANT")
        print("   Bounty #367: Smart Assistant + Wallet Payments")
        print("=" * 70)
        print(f"   💰 Solana: {sol_bal:.4f} SOL (${sol_usd:.2f})")
        if self.eth_wallet:
            eth_bal = self.eth_wallet.get_balance()
            print(f"   💰 Ethereum: {eth_bal:.6f} ETH (${self.price.eth_to_usd(eth_bal):.2f})")
        print(f"   📈 Prices: SOL=${self.price.sol_price:.2f}, ETH=${self.price.eth_price:.2f}")
        print("=" * 70)
        print("   📝 Commands:")
        print("      • 'balance'               - Check all balances")
        print("      • 'send 0.01 SOL to bob'  - Send crypto")
        print("      • 'order coffee'          - Order a product")
        print("      • 'menu'                  - Show products")
        print("      • 'contacts'              - List contacts")
        print("      • 'history'               - Transaction history")
        print("      • 'voice' / 'text'        - Toggle input mode")
        print("      • 'quit'                  - Exit")
        print("=" * 70)
        print("   💳 Supported: SOL, USDC, ETH, USDT")
        print("=" * 70 + "\n")

    def _process(self, text):
        """Route based on state."""
        if self.state == self.STATES['PAYMENT_CONFIRM']:
            return self._handle_payment_confirm(text)
        if self.state == self.STATES['ORDER_CONFIRM']:
            return self._handle_order_confirm(text)
        if self.state == self.STATES['ORDER_PAYMENT']:
            return self._handle_order_payment(text)
        if self.state == self.STATES['CONTACT_ADDRESS']:
            return self._handle_contact_address(text)
        if self.state == self.STATES['CONTACT_NAME']:
            return self._handle_contact_name(text)
        return self._handle_command(text)

    def _handle_command(self, text):
        """Handle normal commands."""
        t = text.lower()

        if any(k in t for k in ['quit', 'exit', 'bye']):
            return {'action': 'quit'}
        if any(k in t for k in ['balance', 'wallet', 'funds']):
            return self._show_balance()
        if any(k in t for k in ['menu', 'products', 'catalog']):
            return self._show_menu()
        if any(k in t for k in ['order', 'buy', 'get me', 'i want']):
            return self._start_order(text)
        if any(k in t for k in ['send', 'pay', 'transfer', 'sol', 'eth', 'usdc', 'usdt']):
            return self._start_payment(text)
        if any(k in t for k in ['contacts', 'address book']):
            return self._show_contacts()
        if 'save contact' in t or 'add contact' in t:
            return self._start_add_contact()
        if 'delete contact' in t:
            return self._delete_contact(text)
        if any(k in t for k in ['history', 'transactions']):
            return self._show_history()
        if 'help' in t:
            return self._show_help()

        self.voice.speak("I didn't understand. Say 'help' for commands.")
        return {'action': 'unknown'}

    def _show_balance(self):
        sol = self.sol_wallet.get_balance()
        usdc = self.sol_wallet.get_usdc_balance()
        
        print(f"\n💰 Balances:")
        print(f"   SOL:  {sol:.4f} (${self.price.sol_to_usd(sol):.2f})")
        print(f"   USDC: {usdc:.2f}")
        
        if self.eth_wallet:
            eth = self.eth_wallet.get_balance()
            usdt = self.eth_wallet.get_usdt_balance()
            print(f"   ETH:  {eth:.6f} (${self.price.eth_to_usd(eth):.2f})")
            print(f"   USDT: {usdt:.2f}")

        self.voice.speak(f"You have {sol:.4f} SOL and {usdc:.2f} USDC.")
        return {'action': 'balance'}

    def _show_menu(self):
        print("\n" + self.store.format_product_list())
        self.voice.speak("We have coffee, tea, pizza, burger, and more. What would you like?")
        return {'action': 'menu'}

    def _start_order(self, text):
        """Start ordering a product."""
        # Extract product name
        products = list(self.store.PRODUCTS.keys())
        product = None
        for p in products:
            if p in text.lower():
                product = p
                break

        if not product:
            self.voice.speak("What would you like to order? Say the product name.")
            print("Available: " + ", ".join(products))
            return {'action': 'order_failed'}

        # Create order
        result = self.store.create_order(product)
        if not result['success']:
            self.voice.speak(result['error'])
            return {'action': 'order_failed'}

        order = result['order']
        self.pending = {'order': order}
        self.state = self.STATES['ORDER_CONFIRM']

        # Show payment options
        print(f"\n📦 Order: {order['product']}")
        print(self.store.format_order_options(order))
        
        self.voice.speak(f"{order['product']} costs ${order['price_usd']:.2f}. Pay with SOL, USDC, ETH, or USDT?")
        return {'action': 'order_created'}

    def _handle_order_confirm(self, text):
        """Handle payment method selection."""
        t = text.upper()
        
        if 'CANCEL' in t or 'NO' in t:
            self.store.cancel_order(self.pending['order']['order_id'])
            self.state = self.STATES['NORMAL']
            self.pending = {}
            self.voice.speak("Order cancelled.")
            return {'action': 'order_cancelled'}

        # Detect payment method
        token = None
        if 'SOL' in t and 'USDC' not in t:
            token = 'SOL'
        elif 'USDC' in t:
            token = 'USDC'
        elif 'ETH' in t and 'USDT' not in t:
            token = 'ETH'
        elif 'USDT' in t:
            token = 'USDT'

        if not token:
            self.voice.speak("Choose SOL, USDC, ETH, or USDT.")
            return {'action': 'awaiting_payment_method'}

        order = self.pending['order']
        amount = order['prices'][token]
        
        self.pending['token'] = token
        self.pending['amount'] = amount
        self.state = self.STATES['ORDER_PAYMENT']

        self.voice.speak(f"Confirm payment of {amount} {token} for {order['product']}? Say yes or no.")
        return {'action': 'payment_selected'}

    def _handle_order_payment(self, text):
        """Process order payment."""
        t = text.lower()

        if any(k in t for k in ['no', 'cancel', 'abort']):
            self.store.cancel_order(self.pending['order']['order_id'])
            self.state = self.STATES['NORMAL']
            self.pending = {}
            self.voice.speak("Order cancelled.")
            return {'action': 'order_cancelled'}

        if any(k in t for k in ['yes', 'confirm', 'ok', 'yeah', 'yep']):
            order = self.pending['order']
            token = self.pending['token']
            amount = self.pending['amount']
            store_address = self.store.get_payment_address(token)

            self.voice.speak(f"Processing payment...")

            # Send payment based on token
            if token in ['SOL', 'USDC']:
                result = self.sol_wallet.send(store_address, amount, token)
            elif token in ['ETH', 'USDT'] and self.eth_wallet:
                result = self.eth_wallet.send(store_address, amount, token)
            else:
                result = {'success': False, 'error': f'{token} wallet not available'}

            if result['success']:
                # Confirm with store
                sig = result.get('signature') or result.get('tx_hash')
                self.store.confirm_payment(order['order_id'], token, sig)
                self.store.complete_order(order['order_id'])

                # Save history
                tx = {
                    'timestamp': datetime.now().isoformat(),
                    'type': 'order',
                    'order_id': order['order_id'],
                    'product': order['product'],
                    'amount': amount,
                    'token': token,
                    'tx': sig,
                    'status': 'success',
                }
                self.tx_history.append(tx)
                Storage.save(HISTORY_FILE, self.tx_history)

                self._print_order_receipt(order, token, amount, sig)
                self.voice.speak(f"Payment successful! Your {order['product']} is ready. Enjoy!")
            else:
                self.voice.speak(f"Payment failed: {result.get('error')}")

            self.state = self.STATES['NORMAL']
            self.pending = {}
            return {'action': 'order_complete' if result['success'] else 'payment_failed'}

        self.voice.speak("Say yes to confirm or no to cancel.")
        return {'action': 'awaiting_confirmation'}

    def _start_payment(self, text):
        """Start a direct payment (not order)."""
        # Parse: send 0.01 SOL to bob
        amount_match = re.search(r'(\d+\.?\d*)', text)
        if not amount_match:
            self.voice.speak("Specify an amount. Example: send 0.01 SOL to bob")
            return {'action': 'payment_failed'}

        amount = float(amount_match.group(1))

        # Detect token
        t = text.upper()
        token = 'SOL'  # default
        if 'USDC' in t:
            token = 'USDC'
        elif 'ETH' in t:
            token = 'ETH'
        elif 'USDT' in t:
            token = 'USDT'

        # Find recipient
        words = text.split()
        recipient = None
        for i, w in enumerate(words):
            if w.lower() == 'to' and i + 1 < len(words):
                recipient = words[i + 1]
                break

        if not recipient:
            self.voice.speak("Specify recipient. Example: send 0.01 SOL to bob")
            return {'action': 'payment_failed'}

        # Resolve address
        recipient_lower = recipient.lower()
        if recipient_lower in self.contacts:
            address = self.contacts[recipient_lower]
            display = recipient_lower.capitalize()
        elif len(recipient) >= 32:
            address = recipient
            display = f"{recipient[:8]}..."
        else:
            self.voice.speak(f"Unknown recipient: {recipient}")
            return {'action': 'payment_failed'}

        self.pending = {
            'amount': amount,
            'token': token,
            'address': address,
            'display': display,
        }
        self.state = self.STATES['PAYMENT_CONFIRM']

        self.voice.speak(f"Send {amount} {token} to {display}? Say yes or no.")
        return {'action': 'payment_pending'}

    def _handle_payment_confirm(self, text):
        """Confirm direct payment."""
        t = text.lower()

        if any(k in t for k in ['no', 'cancel']):
            self.state = self.STATES['NORMAL']
            self.pending = {}
            self.voice.speak("Cancelled.")
            return {'action': 'cancelled'}

        if any(k in t for k in ['yes', 'confirm', 'ok', 'yeah']):
            token = self.pending['token']
            amount = self.pending['amount']
            address = self.pending['address']
            display = self.pending['display']

            self.voice.speak("Processing...")

            if token in ['SOL', 'USDC']:
                result = self.sol_wallet.send(address, amount, token)
            elif token in ['ETH', 'USDT'] and self.eth_wallet:
                result = self.eth_wallet.send(address, amount, token)
            else:
                result = {'success': False, 'error': f'{token} not available'}

            if result['success']:
                sig = result.get('signature') or result.get('tx_hash')
                tx = {
                    'timestamp': datetime.now().isoformat(),
                    'type': 'transfer',
                    'to': display,
                    'amount': amount,
                    'token': token,
                    'tx': sig,
                    'status': 'success',
                }
                self.tx_history.append(tx)
                Storage.save(HISTORY_FILE, self.tx_history)

                self.voice.speak(f"Sent {amount} {token} to {display}!")
                print(f"✅ TX: {sig}")
            else:
                self.voice.speak(f"Failed: {result.get('error')}")

            self.state = self.STATES['NORMAL']
            self.pending = {}
            return {'action': 'payment_done'}

        self.voice.speak("Say yes or no.")
        return {'action': 'awaiting'}

    def _show_contacts(self):
        print("\n📇 Contacts:")
        for name, addr in sorted(self.contacts.items()):
            print(f"   {name.capitalize()}: {addr[:16]}...")
        self.voice.speak(f"You have {len(self.contacts)} contacts.")
        return {'action': 'contacts'}

    def _start_add_contact(self):
        self.state = self.STATES['CONTACT_ADDRESS']
        self.pending = {}
        self.voice.speak("What is the wallet address?")
        return {'action': 'adding_contact'}

    def _handle_contact_address(self, text):
        addr = text.strip()
        if len(addr) < 20:
            self.voice.speak("Invalid address. Try again.")
            return {'action': 'invalid'}
        self.pending['address'] = addr
        self.state = self.STATES['CONTACT_NAME']
        self.voice.speak("What name for this contact?")
        return {'action': 'awaiting_name'}

    def _handle_contact_name(self, text):
        name = text.strip().lower()
        if not name:
            self.voice.speak("Invalid name.")
            return {'action': 'invalid'}
        self.contacts[name] = self.pending['address']
        Storage.save(CONTACTS_FILE, {k: v for k, v in self.contacts.items() if k not in self.DEFAULT_CONTACTS})
        self.state = self.STATES['NORMAL']
        self.pending = {}
        self.voice.speak(f"Contact {name} saved!")
        return {'action': 'contact_saved'}

    def _delete_contact(self, text):
        for word in text.lower().split():
            if word in self.contacts and word not in self.DEFAULT_CONTACTS:
                del self.contacts[word]
                Storage.save(CONTACTS_FILE, {k: v for k, v in self.contacts.items() if k not in self.DEFAULT_CONTACTS})
                self.voice.speak(f"Contact {word} deleted.")
                return {'action': 'deleted'}
        self.voice.speak("Contact not found.")
        return {'action': 'not_found'}

    def _show_history(self):
        if not self.tx_history:
            self.voice.speak("No transactions yet.")
            return {'action': 'history'}

        print("\n📜 History:")
        for tx in reversed(self.tx_history[-10:]):
            if tx['type'] == 'order':
                print(f"   🛒 {tx['product']} - {tx['amount']} {tx['token']}")
            else:
                print(f"   💸 {tx['amount']} {tx['token']} to {tx['to']}")
        self.voice.speak(f"You have {len(self.tx_history)} transactions.")
        return {'action': 'history'}

    def _show_help(self):
        self.voice.speak("You can order products, send crypto, check balance, and manage contacts.")
        return {'action': 'help'}

    def _print_order_receipt(self, order, token, amount, sig):
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    ORDER RECEIPT                              ║
╠══════════════════════════════════════════════════════════════╣
   ✅ Order #{order['order_id']} COMPLETE
   📦 Product: {order['product']}
   💰 Paid: {amount} {token}
   🔗 TX: {sig[:40]}...
╚══════════════════════════════════════════════════════════════╝
""")


def main():
    # Test wallets
    SOL_KEY = '2E9wZuryLrSGkYijwaCsTwMwo953oKbh1zcYDf1Y8X4AWHvVQd6SwzZ5ivt81X5t8rV6Z3yK163dUBCwD8WEkZaF'
    ETH_KEY = None  # Add ETH private key if you have one

    assistant = SmartWalletAssistant(sol_private_key=SOL_KEY, eth_private_key=ETH_KEY)
    assistant.run()


if __name__ == "__main__":
    main()