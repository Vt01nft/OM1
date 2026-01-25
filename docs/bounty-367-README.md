# OM1 Smart Assistant + Wallet Payments

> **Bounty #367** - Enable OM1 to integrate with a smart assistant to help users order products with payment processed through crypto wallets.

[![OM1](https://img.shields.io/badge/OM1-Bounty%20%23367-blue)](https://github.com/OpenmindAGI/OM1/issues/367)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 📋 Overview

This implementation provides a complete smart assistant integration for OM1 that enables:

- **Voice & Text Interaction** - Natural language interface with speech recognition and text-to-speech
- **Multi-Chain Wallet Payments** - Solana (SOL, USDC) and Ethereum (ETH, USDT) support
- **Product Ordering** - Order coffee, tea, pizza, burgers with crypto payments
- **Contact Management** - Save and manage wallet contacts
- **Transaction History** - Track all orders and transfers

## 🎯 Features

### ✅ Bounty Requirements Met

| Requirement | Status | Implementation |
|------------|--------|----------------|
| OM1 communicates with assistant | ✅ | SmartAssistantProvider with voice/text modes |
| Place orders or trigger actions | ✅ | StoreAssistantProvider handles product ordering |
| Wallet-based payments | ✅ | WalletPaymentProvider with Solana + Ethereum |
| Secure crypto wallet execution | ✅ | Real blockchain transactions on testnets |
| Transaction status reporting | ✅ | Success/pending/failed status with history |
| Full workflow demonstration | ✅ | Order → Assistant → Payment → Confirmation |

### 🆕 Additional Features

- **Friendly Conversation** - Greetings, thank you, how are you, jokes
- **Multi-Token Support** - SOL, ETH, USDC, USDT
- **Live Price Feeds** - Real-time prices from CoinGecko API
- **Smart Payment Suggestions** - Suggests alternative tokens if balance insufficient
- **Voice Fallback** - Text input when voice recognition fails

## 📁 Project Structure

```
bounty-367/
├── src/
│   └── providers/
│       ├── smart_assistant_provider.py   # Voice/text interface
│       ├── wallet_payment_provider.py    # Multi-chain payments
│       ├── store_assistant_provider.py   # Product catalog & orders
│       ├── solana_wallet_provider.py     # Solana blockchain
│       └── ethereum_wallet_provider.py   # Ethereum blockchain
├── demo_smart_wallet.py                  # Interactive demo
├── tests/
│   └── test_smart_assistant.py           # Unit tests
├── README.md                             # This file
└── requirements.txt                      # Dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- OM1 repository cloned
- Microphone (for voice mode)

### Installation

```bash
# Clone OM1 repository (if not already)
git clone https://github.com/OpenmindAGI/OM1.git
cd OM1

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install additional dependencies for this bounty
pip install pyttsx3 SpeechRecognition solana solders web3
```

### Running the Demo

```bash
python demo_smart_wallet.py
```

## 💡 Usage

### Voice Mode

```
🎤 "Hello" / "How are you?"
🎤 "Check my balance"
🎤 "Order a coffee"
🎤 "Show my wallets"
🎤 "Send 0.01 SOL to Bob"
🎤 "Thank you"
🎤 "Quit"
```

### Text Mode

```
> balance
> balance in sol
> order coffee
> wallets
> send 0.01 sol to bob
> contacts
> history
> quit
```

## 🔧 API Reference

### SmartAssistantProvider

The main interface for voice and text interactions.

```python
from providers.smart_assistant_provider import SmartAssistantProvider

# Initialize
assistant = SmartAssistantProvider(user_name="Alice")

# Enable voice mode
assistant.set_voice_mode(True)

# Process commands
result = assistant.process_command("check my balance")
# Returns: {'intent': 'balance', 'action': 'check_balance', ...}

# Output with voice
assistant.output("Hello! How can I help?")

# Get input (voice or text based on mode)
response = assistant.get_input("What would you like to order?")

# Conversation helpers
assistant.greet()        # "Hello Alice! How can I help?"
assistant.thank()        # "You're welcome!"
assistant.goodbye()      # "Goodbye Alice!"
```

### WalletPaymentProvider

Unified multi-chain wallet management and payments.

```python
from providers.wallet_payment_provider import WalletPaymentProvider

# Initialize
wallet = WalletPaymentProvider(
    solana_network='devnet',
    ethereum_network='sepolia'
)

# Check balances
balances = wallet.get_all_balances()
# {'SOL': 4.81, 'USDC': 0.0, 'ETH': 0.49, 'USDT': 0.0}

# Get balances with USD values
balances_usd = wallet.get_balances_with_usd()

# Calculate payment in multiple tokens
prices = wallet.calculate_payment(usd_amount=5.50)
# {'SOL': 0.0393, 'ETH': 0.0016, 'USDC': 5.50, 'USDT': 5.50}

# Make payment
result = wallet.pay(
    token='SOL',
    amount=0.0393,
    recipient='store_address',
    description='Order: Latte'
)

# Pay to store
result = wallet.pay_store(token='SOL', amount=0.0393, item='Latte')

# Manage contacts
wallet.add_contact('bob', 'BobsWalletAddress123...', 'SOL')
wallet.pay_contact('bob', amount=0.1, token='SOL')

# Transaction history
history = wallet.get_history(limit=10)
```

### StoreAssistantProvider

Product catalog and order management.

```python
from providers.store_assistant_provider import StoreAssistantProvider

# Initialize
store = StoreAssistantProvider()

# Get products
products = store.get_products()  # ['coffee', 'tea', 'pizza', 'burger']

# Get varieties
varieties = store.get_varieties('coffee')
# [('Espresso', 4.00), ('Americano', 4.50), ('Latte', 5.50), ...]

# Create order
order = store.create_order('coffee', 'latte')
# {'product': 'coffee', 'variety': 'Latte', 'price_usd': 5.50, ...}

# Process payment
result = store.process_payment(order, 'SOL', 0.0393, wallet_provider)
```

## 🔄 Workflow

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│                 │     │                  │     │                 │
│  User Request   │────▶│  Smart Assistant │────▶│  Store Provider │
│  "Order coffee" │     │  (Voice/Text)    │     │  (Catalog)      │
│                 │     │                  │     │                 │
└─────────────────┘     └────────┬─────────┘     └────────┬────────┘
                                 │                        │
                                 │ Select variety         │ Create order
                                 │ Choose payment         │
                                 ▼                        ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│                 │     │                  │     │                 │
│  Confirmation   │◀────│  Wallet Provider │◀────│  Payment        │
│  "On its way!"  │     │  (Multi-chain)   │     │  Processing     │
│                 │     │                  │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## 🧪 Testing

```bash
# Run tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_smart_assistant.py -v
```

## 🌐 Networks

### Solana (Devnet)
- **Faucet**: https://faucet.solana.com
- **Explorer**: https://explorer.solana.com/?cluster=devnet

### Ethereum (Sepolia)
- **Faucet**: https://sepoliafaucet.com
- **Explorer**: https://sepolia.etherscan.io

## 📸 Screenshots

### Main Menu
```
════════════════════════════════════════════════════════════
   🚀 OM1 SMART WALLET v3.5
   Bounty #367
════════════════════════════════════════════════════════════

   [1] 🎤 VOICE - I'll speak all responses!
   [2] ⌨️  TEXT  - Type commands
   [3] 🚪 EXIT
```

### Balance Check
```
   🤖 You have 4.7788 SOL worth $601.60, 0.495503 ETH worth
      $1451.30, 0.00 USDC, and 0.00 USDT.
```

### Order Flow
```
   🤖 Great! For pizza we have: Margherita, Pepperoni, Veggie,
      Hawaiian. Which one?
   👤 You: pepperoni
   🤖 Great! Pepperoni costs $16.00.
   🤖 That's 0.1271 SOL or 0.005464 ETH. Pay with SOL, ETH,
      USDC, or USDT?
   👤 You: solana
   🤖 Confirm 0.1271 SOL for Pepperoni? Say yes or no.
   👤 You: yes
   🤖 Processing payment...
   🤖 Payment confirmed! Your Pepperoni is on its way! Enjoy!
```

## 🎥 Demo Video

[Watch the demo video](https://youtu.be/your-video-link)

*Shows complete workflow: voice interaction → order placement → crypto payment → confirmation*

## 🐦 Social

Follow and tag [@opaborobots](https://twitter.com/opaborobots) on Twitter!

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

- **GitHub Issues**: [OM1 Issues](https://github.com/OpenmindAGI/OM1/issues)
- **Bounty**: [Issue #367](https://github.com/OpenmindAGI/OM1/issues/367)

---

**Built with ❤️ for the OM1 community**
