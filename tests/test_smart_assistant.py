"""Unit Tests for Smart Assistant + Wallet Payments.

Bounty #367: OM1 + Smart Assistant + Wallet Payments

Tests for:
- SmartAssistantProvider
- WalletPaymentProvider
- StoreAssistantProvider
"""

import os
import sys
import tempfile

import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ═══════════════════════════════════════════════════════════════════
# MOCK PROVIDERS (for testing without blockchain)
# ═══════════════════════════════════════════════════════════════════


class MockSolanaWallet:
    """Mock Solana wallet for testing."""

    def __init__(self, balance=5.0, usdc_balance=10.0):
        """Initialize mock wallet."""
        self.wallet_address = "MockSolanaAddress123456789"
        self._balance = balance
        self._usdc_balance = usdc_balance

    def get_balance(self):
        """Get SOL balance."""
        return self._balance

    def get_usdc_balance(self):
        """Get USDC balance."""
        return self._usdc_balance

    def send(self, to, amount, token="SOL"):
        """Send tokens."""
        if token == "SOL" and amount > self._balance:
            return {"success": False, "error": "Insufficient SOL"}
        if token == "USDC" and amount > self._usdc_balance:
            return {"success": False, "error": "Insufficient USDC"}

        if token == "SOL":
            self._balance -= amount
        else:
            self._usdc_balance -= amount

        return {"success": True, "signature": "mock_tx_123"}


class MockEthereumWallet:
    """Mock Ethereum wallet for testing."""

    def __init__(self, balance=0.5, usdt_balance=20.0):
        """Initialize mock wallet."""
        self.wallet_address = "0xMockEthereumAddress123456789"
        self._balance = balance
        self._usdt_balance = usdt_balance

    def get_balance(self):
        """Get ETH balance."""
        return self._balance

    def get_usdt_balance(self):
        """Get USDT balance."""
        return self._usdt_balance

    def send(self, to, amount, token="ETH"):
        """Send tokens."""
        if token == "ETH" and amount > self._balance:
            return {"success": False, "error": "Insufficient ETH"}
        if token == "USDT" and amount > self._usdt_balance:
            return {"success": False, "error": "Insufficient USDT"}

        if token == "ETH":
            self._balance -= amount
        else:
            self._usdt_balance -= amount

        return {"success": True, "tx_hash": "mock_tx_456"}


# ═══════════════════════════════════════════════════════════════════
# SMART ASSISTANT TESTS
# ═══════════════════════════════════════════════════════════════════


class TestSmartAssistantProvider:
    """Tests for SmartAssistantProvider."""

    def test_init(self):
        """Test initialization."""
        from providers.smart_assistant_provider import SmartAssistantProvider

        assistant = SmartAssistantProvider(user_name="TestUser", voice_enabled=False)

        assert assistant.user_name == "TestUser"
        assert not assistant.voice_mode

    def test_set_user_name(self):
        """Test user name setting."""
        from providers.smart_assistant_provider import SmartAssistantProvider

        assistant = SmartAssistantProvider(voice_enabled=False)
        assistant.set_user_name("Alice")

        assert assistant.user_name == "Alice"

    def test_set_voice_mode(self):
        """Test voice mode toggle."""
        from providers.smart_assistant_provider import SmartAssistantProvider

        assistant = SmartAssistantProvider(voice_enabled=False)

        assistant.set_voice_mode(True)
        assert assistant.voice_mode

        assistant.set_voice_mode(False)
        assert not assistant.voice_mode

    def test_detect_token_sol(self):
        """Test SOL token detection."""
        from providers.smart_assistant_provider import SmartAssistantProvider

        assistant = SmartAssistantProvider(voice_enabled=False)

        assert assistant._detect_token("pay with solana") == "SOL"
        assert assistant._detect_token("use sol") == "SOL"
        assert assistant._detect_token("balance in SOL") == "SOL"

    def test_detect_token_eth(self):
        """Test ETH token detection."""
        from providers.smart_assistant_provider import SmartAssistantProvider

        assistant = SmartAssistantProvider(voice_enabled=False)

        assert assistant._detect_token("pay with ethereum") == "ETH"
        assert assistant._detect_token("use eth") == "ETH"

    def test_detect_token_stablecoins(self):
        """Test stablecoin detection."""
        from providers.smart_assistant_provider import SmartAssistantProvider

        assistant = SmartAssistantProvider(voice_enabled=False)

        assert assistant._detect_token("pay with usdc") == "USDC"
        assert assistant._detect_token("use usdt") == "USDT"

    def test_detect_product(self):
        """Test product detection."""
        from providers.smart_assistant_provider import SmartAssistantProvider

        assistant = SmartAssistantProvider(voice_enabled=False)

        assert assistant._detect_product("order a coffee") == "coffee"
        assert assistant._detect_product("get me pizza") == "pizza"
        assert assistant._detect_product("i want tea") == "tea"
        assert assistant._detect_product("buy burger") == "burger"

    def test_process_greeting(self):
        """Test greeting command processing."""
        from providers.smart_assistant_provider import SmartAssistantProvider

        assistant = SmartAssistantProvider(user_name="Test", voice_enabled=False)

        result = assistant.process_command("hello")
        assert result["intent"] == "greeting"
        assert "Test" in result["response"]

    def test_process_balance(self):
        """Test balance command processing."""
        from providers.smart_assistant_provider import SmartAssistantProvider

        assistant = SmartAssistantProvider(voice_enabled=False)

        result = assistant.process_command("check my balance")
        assert result["intent"] == "balance"
        assert result["action"] == "check_balance"

    def test_process_order(self):
        """Test order command processing."""
        from providers.smart_assistant_provider import SmartAssistantProvider

        assistant = SmartAssistantProvider(voice_enabled=False)

        result = assistant.process_command("order a coffee")
        assert result["intent"] == "order"
        assert result["action"] == "start_order"
        assert result["data"]["product"] == "coffee"

    def test_process_thanks(self):
        """Test thank you command processing."""
        from providers.smart_assistant_provider import SmartAssistantProvider

        assistant = SmartAssistantProvider(voice_enabled=False)

        result = assistant.process_command("thank you")
        assert result["intent"] == "thanks"

    def test_process_goodbye(self):
        """Test goodbye command processing."""
        from providers.smart_assistant_provider import SmartAssistantProvider

        assistant = SmartAssistantProvider(voice_enabled=False)

        result = assistant.process_command("goodbye")
        assert result["intent"] == "goodbye"

    def test_process_help(self):
        """Test help command processing."""
        from providers.smart_assistant_provider import SmartAssistantProvider

        assistant = SmartAssistantProvider(voice_enabled=False)

        result = assistant.process_command("help")
        assert result["intent"] == "help"

    def test_status(self):
        """Test status reporting."""
        from providers.smart_assistant_provider import SmartAssistantProvider

        assistant = SmartAssistantProvider(user_name="Test", voice_enabled=False)

        status = assistant.get_status()
        assert status["user_name"] == "Test"
        assert not status["voice_mode"]


# ═══════════════════════════════════════════════════════════════════
# STORE ASSISTANT TESTS
# ═══════════════════════════════════════════════════════════════════


class TestStoreAssistantProvider:
    """Tests for StoreAssistantProvider."""

    def test_init(self):
        """Test initialization."""
        from providers.store_assistant_provider import StoreAssistantProvider

        store = StoreAssistantProvider()
        assert store.current_order is None

    def test_get_products(self):
        """Test getting product list."""
        from providers.store_assistant_provider import StoreAssistantProvider

        store = StoreAssistantProvider()
        products = store.get_products()

        assert "coffee" in products
        assert "tea" in products
        assert "pizza" in products
        assert "burger" in products

    def test_get_varieties(self):
        """Test getting product varieties."""
        from providers.store_assistant_provider import StoreAssistantProvider

        store = StoreAssistantProvider()

        varieties = store.get_varieties("coffee")
        variety_names = [name for name, price in varieties]

        assert "Espresso" in variety_names
        assert "Latte" in variety_names

    def test_find_variety(self):
        """Test variety matching."""
        from providers.store_assistant_provider import StoreAssistantProvider

        store = StoreAssistantProvider()

        result = store.find_variety("coffee", "latte")
        assert result is not None
        assert result[0] == "Latte"
        assert result[1] == 5.50

    def test_find_variety_partial(self):
        """Test partial variety matching."""
        from providers.store_assistant_provider import StoreAssistantProvider

        store = StoreAssistantProvider()

        result = store.find_variety("pizza", "marg")
        assert result is not None
        assert "Margherita" in result[0]

    def test_detect_product(self):
        """Test product detection from text."""
        from providers.store_assistant_provider import StoreAssistantProvider

        store = StoreAssistantProvider()

        assert store.detect_product("i want coffee") == "coffee"
        assert store.detect_product("order pizza please") == "pizza"

    def test_create_order(self):
        """Test order creation."""
        from providers.store_assistant_provider import StoreAssistantProvider

        store = StoreAssistantProvider()

        order = store.create_order("coffee", "espresso")

        assert order is not None
        assert order["variety"] == "Espresso"
        assert order["price_usd"] == 4.00
        assert order["status"] == "pending"

    def test_calculate_prices(self):
        """Test price calculation."""
        from providers.store_assistant_provider import StoreAssistantProvider

        store = StoreAssistantProvider()

        prices = store.calculate_prices(10.00, sol_price=100.0, eth_price=2000.0)

        assert prices["SOL"] == 0.1
        assert prices["ETH"] == 0.005
        assert prices["USDC"] == 10.00
        assert prices["USDT"] == 10.00

    def test_cancel_order(self):
        """Test order cancellation."""
        from providers.store_assistant_provider import StoreAssistantProvider

        store = StoreAssistantProvider()
        store.create_order("coffee", "latte")

        store.cancel_order()

        assert store.current_order is None

    def test_status(self):
        """Test status reporting."""
        from providers.store_assistant_provider import StoreAssistantProvider

        store = StoreAssistantProvider()

        status = store.get_status()
        assert status["products_available"] == 4


# ═══════════════════════════════════════════════════════════════════
# WALLET PAYMENT TESTS
# ═══════════════════════════════════════════════════════════════════


class TestWalletPaymentProvider:
    """Tests for WalletPaymentProvider (with mocks)."""

    def test_price_service(self):
        """Test price service calculation."""
        from providers.wallet_payment_provider import PriceService

        price = PriceService()

        # Test conversion
        assert price.usd_to_sol(140.0) == pytest.approx(1.0, rel=0.5)
        assert price.sol_to_usd(1.0) == pytest.approx(140.0, rel=0.5)

    def test_calculate_payment(self):
        """Test payment calculation."""
        from providers.wallet_payment_provider import WalletPaymentProvider

        # Create with mock setup
        wallet = WalletPaymentProvider.__new__(WalletPaymentProvider)
        wallet.price = type(
            "Price",
            (),
            {
                "sol_price": 100.0,
                "eth_price": 2000.0,
                "usd_to_sol": lambda self, x: x / 100,
                "usd_to_eth": lambda self, x: x / 2000,
            },
        )()

        prices = wallet.calculate_payment(10.00)

        assert prices["SOL"] == 0.1
        assert prices["ETH"] == 0.005
        assert prices["USDC"] == 10.00
        assert prices["USDT"] == 10.00

    def test_get_balance_with_mock(self):
        """Test balance retrieval with mock wallet."""
        from providers.wallet_payment_provider import WalletPaymentProvider

        # Create provider with mocks
        wallet = WalletPaymentProvider.__new__(WalletPaymentProvider)
        wallet.solana_wallet = MockSolanaWallet(balance=5.0)
        wallet.ethereum_wallet = MockEthereumWallet(balance=0.5)

        assert wallet.get_balance("SOL") == 5.0
        assert wallet.get_balance("ETH") == 0.5

    def test_can_afford(self):
        """Test affordability check."""
        from providers.wallet_payment_provider import WalletPaymentProvider

        wallet = WalletPaymentProvider.__new__(WalletPaymentProvider)
        wallet.solana_wallet = MockSolanaWallet(balance=5.0)
        wallet.ethereum_wallet = MockEthereumWallet(balance=0.5)

        assert wallet.can_afford("SOL", 4.0)
        assert not wallet.can_afford("SOL", 10.0)
        assert wallet.can_afford("ETH", 0.3)
        assert not wallet.can_afford("ETH", 1.0)

    def test_find_affordable_token(self):
        """Test finding affordable token."""
        from providers.wallet_payment_provider import WalletPaymentProvider

        wallet = WalletPaymentProvider.__new__(WalletPaymentProvider)
        wallet.solana_wallet = MockSolanaWallet(balance=0.1, usdc_balance=100.0)
        wallet.ethereum_wallet = MockEthereumWallet(balance=0.01, usdt_balance=50.0)

        amounts = {"SOL": 1.0, "ETH": 0.5, "USDC": 10.0, "USDT": 10.0}

        result = wallet.find_affordable_token(amounts)
        assert result == "USDC"  # Only USDC has enough

    def test_contact_management(self):
        """Test contact CRUD operations."""
        from providers.wallet_payment_provider import WalletPaymentProvider

        # Create temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        temp_file.close()

        try:
            wallet = WalletPaymentProvider.__new__(WalletPaymentProvider)
            wallet.contacts = {}
            wallet.contacts_file = temp_file.name
            wallet._save_json = lambda p, d: True

            # Add contact
            wallet.add_contact("bob", "BobAddress123", "SOL")
            assert "bob" in wallet.contacts

            # Get contact
            contact = wallet.get_contact("bob")
            assert contact["address"] == "BobAddress123"

            # Remove contact
            wallet.remove_contact("bob")
            assert "bob" not in wallet.contacts
        finally:
            os.unlink(temp_file.name)


# ═══════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════


class TestIntegration:
    """Integration tests for the full workflow."""

    def test_order_workflow(self):
        """Test complete order workflow."""
        from providers.smart_assistant_provider import SmartAssistantProvider
        from providers.store_assistant_provider import StoreAssistantProvider

        # Setup
        assistant = SmartAssistantProvider(user_name="Test", voice_enabled=False)
        store = StoreAssistantProvider()

        # Process order command
        result = assistant.process_command("order a coffee")
        assert result["intent"] == "order"

        # Create order
        order = store.create_order("coffee", "latte")
        assert order is not None
        assert order["variety"] == "Latte"

        # Calculate prices
        prices = store.calculate_prices(order["price_usd"], 100.0, 2000.0)
        assert "SOL" in prices
        assert "ETH" in prices

    def test_balance_check_workflow(self):
        """Test balance check workflow."""
        from providers.smart_assistant_provider import SmartAssistantProvider

        assistant = SmartAssistantProvider(voice_enabled=False)

        # Process balance command with token
        result = assistant.process_command("check my balance in sol")
        assert result["intent"] == "balance"
        assert result["data"]["token"] == "SOL"

    def test_send_workflow(self):
        """Test send command workflow."""
        from providers.smart_assistant_provider import SmartAssistantProvider

        assistant = SmartAssistantProvider(voice_enabled=False)

        result = assistant.process_command("send tokens")
        assert result["intent"] == "send"
        assert result["action"] == "send_tokens"


# ═══════════════════════════════════════════════════════════════════
# RUN TESTS
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
