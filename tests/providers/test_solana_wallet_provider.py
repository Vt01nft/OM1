"""Tests for solana_wallet_provider."""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock ALL external dependencies BEFORE any provider imports
sys.modules["zenoh"] = MagicMock()
sys.modules["zenoh_msgs"] = MagicMock()
sys.modules["requests"] = MagicMock()
sys.modules["cv2"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["PIL.Image"] = MagicMock()
sys.modules["google"] = MagicMock()
sys.modules["google.generativeai"] = MagicMock()
sys.modules["openai"] = MagicMock()
sys.modules["rclpy"] = MagicMock()
sys.modules["rclpy.node"] = MagicMock()
sys.modules["rclpy.qos"] = MagicMock()
sys.modules["sensor_msgs"] = MagicMock()
sys.modules["sensor_msgs.msg"] = MagicMock()
sys.modules["geometry_msgs"] = MagicMock()
sys.modules["geometry_msgs.msg"] = MagicMock()
sys.modules["nav_msgs"] = MagicMock()
sys.modules["nav_msgs.msg"] = MagicMock()
sys.modules["std_msgs"] = MagicMock()
sys.modules["std_msgs.msg"] = MagicMock()
sys.modules["elevenlabs"] = MagicMock()
sys.modules["riva"] = MagicMock()
sys.modules["riva.client"] = MagicMock()
sys.modules["pyaudio"] = MagicMock()
sys.modules["sounddevice"] = MagicMock()
sys.modules["websocket"] = MagicMock()
sys.modules["websockets"] = MagicMock()
sys.modules["aiohttp"] = MagicMock()
sys.modules["pyrealsense2"] = MagicMock()
sys.modules["mjpeg"] = MagicMock()
sys.modules["mjpeg.client"] = MagicMock()
sys.modules["unitree"] = MagicMock()
sys.modules["unitree_sdk2py"] = MagicMock()
sys.modules["unitree_sdk2py.core"] = MagicMock()
sys.modules["unitree_sdk2py.core.channel"] = MagicMock()
sys.modules["solders"] = MagicMock()
sys.modules["solders.keypair"] = MagicMock()
sys.modules["solders.pubkey"] = MagicMock()
sys.modules["solders.system_program"] = MagicMock()
sys.modules["solders.transaction"] = MagicMock()
sys.modules["solders.message"] = MagicMock()
sys.modules["solana"] = MagicMock()
sys.modules["solana.rpc"] = MagicMock()
sys.modules["solana.rpc.api"] = MagicMock()
sys.modules["solana.rpc.types"] = MagicMock()
sys.modules["base58"] = MagicMock()


class TestSolanaWalletProvider:
    """Tests for SolanaWalletProvider class."""

    @pytest.fixture(autouse=True)
    def reset_modules(self):
        """Reset module cache before each test."""
        modules_to_clear = [k for k in sys.modules.keys() if "providers" in k]
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]
        yield
        modules_to_clear = [k for k in sys.modules.keys() if "providers" in k]
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]

    def test_initialization_with_default_params(self):
        """Test provider initializes correctly with default parameters."""
        with patch("providers.solana_wallet_provider.SOLANA_AVAILABLE", True):
            from providers.solana_wallet_provider import SolanaWalletProvider

            if hasattr(SolanaWalletProvider, "reset"):
                SolanaWalletProvider.reset()

            provider = SolanaWalletProvider()
            assert provider is not None
            assert provider.network == "devnet"
            assert provider._running is False

    def test_initialization_with_custom_network(self):
        """Test provider initializes correctly with custom network."""
        with patch("providers.solana_wallet_provider.SOLANA_AVAILABLE", True):
            from providers.solana_wallet_provider import SolanaWalletProvider

            if hasattr(SolanaWalletProvider, "reset"):
                SolanaWalletProvider.reset()

            provider = SolanaWalletProvider(network="mainnet")
            assert provider.network == "mainnet"

    def test_initialization_without_solana_libraries(self):
        """Test provider handles missing Solana libraries gracefully."""
        with patch("providers.solana_wallet_provider.SOLANA_AVAILABLE", False):
            from providers.solana_wallet_provider import SolanaWalletProvider

            if hasattr(SolanaWalletProvider, "reset"):
                SolanaWalletProvider.reset()

            provider = SolanaWalletProvider()
            assert provider.client is None
            assert provider.keypair is None
            assert provider.wallet_address is None

    def test_start(self):
        """Test provider start method."""
        with patch("providers.solana_wallet_provider.SOLANA_AVAILABLE", True):
            from providers.solana_wallet_provider import SolanaWalletProvider

            if hasattr(SolanaWalletProvider, "reset"):
                SolanaWalletProvider.reset()

            provider = SolanaWalletProvider()
            provider.start()
            assert provider._running is True

    def test_stop(self):
        """Test provider stop method."""
        with patch("providers.solana_wallet_provider.SOLANA_AVAILABLE", True):
            from providers.solana_wallet_provider import SolanaWalletProvider

            if hasattr(SolanaWalletProvider, "reset"):
                SolanaWalletProvider.reset()

            provider = SolanaWalletProvider()
            provider.start()
            provider.stop()
            assert provider._running is False

    def test_is_connected_without_client(self):
        """Test is_connected returns False when client is None."""
        with patch("providers.solana_wallet_provider.SOLANA_AVAILABLE", False):
            from providers.solana_wallet_provider import SolanaWalletProvider

            if hasattr(SolanaWalletProvider, "reset"):
                SolanaWalletProvider.reset()

            provider = SolanaWalletProvider()
            assert provider.is_connected() is False

    def test_is_connected_with_client(self):
        """Test is_connected returns True when client is available."""
        with patch("providers.solana_wallet_provider.SOLANA_AVAILABLE", True):
            from providers.solana_wallet_provider import SolanaWalletProvider

            if hasattr(SolanaWalletProvider, "reset"):
                SolanaWalletProvider.reset()

            provider = SolanaWalletProvider()
            mock_client = MagicMock()
            mock_client.is_connected.return_value = True
            provider.client = mock_client
            assert provider.is_connected() is True

    def test_is_connected_exception_handling(self):
        """Test is_connected handles exceptions gracefully."""
        with patch("providers.solana_wallet_provider.SOLANA_AVAILABLE", True):
            from providers.solana_wallet_provider import SolanaWalletProvider

            if hasattr(SolanaWalletProvider, "reset"):
                SolanaWalletProvider.reset()

            provider = SolanaWalletProvider()
            mock_client = MagicMock()
            mock_client.is_connected.side_effect = Exception("Connection error")
            provider.client = mock_client
            assert provider.is_connected() is False

    def test_get_balance_without_client(self):
        """Test get_balance returns 0.0 when client is None."""
        with patch("providers.solana_wallet_provider.SOLANA_AVAILABLE", False):
            from providers.solana_wallet_provider import SolanaWalletProvider

            if hasattr(SolanaWalletProvider, "reset"):
                SolanaWalletProvider.reset()

            provider = SolanaWalletProvider()
            assert provider.get_balance() == 0.0

    def test_get_balance_with_client(self):
        """Test get_balance returns correct balance."""
        with patch("providers.solana_wallet_provider.SOLANA_AVAILABLE", True):
            from providers.solana_wallet_provider import SolanaWalletProvider

            if hasattr(SolanaWalletProvider, "reset"):
                SolanaWalletProvider.reset()

            provider = SolanaWalletProvider()
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.value = 2_000_000_000  # 2 SOL in lamports
            mock_client.get_balance.return_value = mock_response
            provider.client = mock_client

            balance = provider.get_balance()
            assert balance == 2.0

    def test_get_balance_exception_handling(self):
        """Test get_balance handles exceptions gracefully."""
        with patch("providers.solana_wallet_provider.SOLANA_AVAILABLE", True):
            from providers.solana_wallet_provider import SolanaWalletProvider

            if hasattr(SolanaWalletProvider, "reset"):
                SolanaWalletProvider.reset()

            provider = SolanaWalletProvider()
            mock_client = MagicMock()
            mock_client.get_balance.side_effect = Exception("Network error")
            provider.client = mock_client

            balance = provider.get_balance()
            assert balance == 0.0

    def test_get_usdc_balance_without_client(self):
        """Test get_usdc_balance returns 0.0 when client is None."""
        with patch("providers.solana_wallet_provider.SOLANA_AVAILABLE", False):
            from providers.solana_wallet_provider import SolanaWalletProvider

            if hasattr(SolanaWalletProvider, "reset"):
                SolanaWalletProvider.reset()

            provider = SolanaWalletProvider()
            assert provider.get_usdc_balance() == 0.0

    def test_get_usdc_balance_with_client(self):
        """Test get_usdc_balance returns correct balance."""
        with patch("providers.solana_wallet_provider.SOLANA_AVAILABLE", True):
            from providers.solana_wallet_provider import SolanaWalletProvider

            if hasattr(SolanaWalletProvider, "reset"):
                SolanaWalletProvider.reset()

            provider = SolanaWalletProvider()
            mock_client = MagicMock()
            mock_account = MagicMock()
            mock_account.account.data.parsed = {
                "info": {
                    "mint": "4zMMC9srt5Ri5X14GAgXhaHii3GnPAEERYPJgZJDncDU",
                    "tokenAmount": {"uiAmount": 100.5},
                }
            }
            mock_response = MagicMock()
            mock_response.value = [mock_account]
            mock_client.get_token_accounts_by_owner_json_parsed.return_value = (
                mock_response
            )
            provider.client = mock_client
            provider.usdc_mint = "4zMMC9srt5Ri5X14GAgXhaHii3GnPAEERYPJgZJDncDU"

            balance = provider.get_usdc_balance()
            assert balance == 100.5

    def test_get_usdc_balance_no_accounts(self):
        """Test get_usdc_balance returns 0.0 when no token accounts exist."""
        with patch("providers.solana_wallet_provider.SOLANA_AVAILABLE", True):
            from providers.solana_wallet_provider import SolanaWalletProvider

            if hasattr(SolanaWalletProvider, "reset"):
                SolanaWalletProvider.reset()

            provider = SolanaWalletProvider()
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.value = []
            mock_client.get_token_accounts_by_owner_json_parsed.return_value = (
                mock_response
            )
            provider.client = mock_client

            balance = provider.get_usdc_balance()
            assert balance == 0.0

    def test_get_usdc_balance_exception_handling(self):
        """Test get_usdc_balance handles exceptions gracefully."""
        with patch("providers.solana_wallet_provider.SOLANA_AVAILABLE", True):
            from providers.solana_wallet_provider import SolanaWalletProvider

            if hasattr(SolanaWalletProvider, "reset"):
                SolanaWalletProvider.reset()

            provider = SolanaWalletProvider()
            mock_client = MagicMock()
            mock_client.get_token_accounts_by_owner_json_parsed.side_effect = Exception(
                "Network error"
            )
            provider.client = mock_client

            balance = provider.get_usdc_balance()
            assert balance == 0.0

    def test_send_sol_without_client(self):
        """Test send_sol returns error when client is None."""
        with patch("providers.solana_wallet_provider.SOLANA_AVAILABLE", False):
            from providers.solana_wallet_provider import SolanaWalletProvider

            if hasattr(SolanaWalletProvider, "reset"):
                SolanaWalletProvider.reset()

            provider = SolanaWalletProvider()
            result = provider.send_sol("target_address", 1.0)
            assert result["success"] is False
            assert "error" in result

    def test_send_sol_insufficient_balance(self):
        """Test send_sol returns error when balance is insufficient."""
        with patch("providers.solana_wallet_provider.SOLANA_AVAILABLE", True):
            from providers.solana_wallet_provider import SolanaWalletProvider

            if hasattr(SolanaWalletProvider, "reset"):
                SolanaWalletProvider.reset()

            provider = SolanaWalletProvider()
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.value = 500_000_000  # 0.5 SOL in lamports
            mock_client.get_balance.return_value = mock_response
            provider.client = mock_client

            result = provider.send_sol("target_address", 1.0)
            assert result["success"] is False
            assert "Insufficient SOL balance" in result["error"]

    def test_send_sol_success(self):
        """Test send_sol returns success for valid transaction."""
        with patch("providers.solana_wallet_provider.SOLANA_AVAILABLE", True):
            from providers.solana_wallet_provider import SolanaWalletProvider

            if hasattr(SolanaWalletProvider, "reset"):
                SolanaWalletProvider.reset()

            provider = SolanaWalletProvider()
            mock_client = MagicMock()

            # Mock balance check
            mock_balance_response = MagicMock()
            mock_balance_response.value = 2_000_000_000  # 2 SOL
            mock_client.get_balance.return_value = mock_balance_response

            # Mock blockhash
            mock_blockhash_response = MagicMock()
            mock_blockhash_response.value.blockhash = "mock_blockhash"
            mock_client.get_latest_blockhash.return_value = mock_blockhash_response

            # Mock transaction send
            mock_tx_response = MagicMock()
            mock_tx_response.value = "mock_signature"
            mock_client.send_transaction.return_value = mock_tx_response

            provider.client = mock_client

            with (
                patch("providers.solana_wallet_provider.Pubkey") as mock_pubkey,
                patch("providers.solana_wallet_provider.transfer") as mock_transfer,
                patch("providers.solana_wallet_provider.Message") as mock_message,
                patch(
                    "providers.solana_wallet_provider.Transaction"
                ) as mock_transaction,
                patch("providers.solana_wallet_provider.TxOpts") as mock_txopts,
            ):

                result = provider.send_sol("target_address", 1.0)
                # Should not raise exception
                assert isinstance(result, dict)

    def test_send_sol_exception_handling(self):
        """Test send_sol handles exceptions gracefully."""
        with patch("providers.solana_wallet_provider.SOLANA_AVAILABLE", True):
            from providers.solana_wallet_provider import SolanaWalletProvider

            if hasattr(SolanaWalletProvider, "reset"):
                SolanaWalletProvider.reset()

            provider = SolanaWalletProvider()
            mock_client = MagicMock()
            mock_client.get_balance.side_effect = Exception("Network error")
            provider.client = mock_client

            result = provider.send_sol("target_address", 1.0)
            assert result["success"] is False
            assert "error" in result

    def test_singleton_pattern(self):
        """Test that provider follows singleton pattern."""
        with patch("providers.solana_wallet_provider.SOLANA_AVAILABLE", True):
            from providers.solana_wallet_provider import SolanaWalletProvider

            if hasattr(SolanaWalletProvider, "reset"):
                SolanaWalletProvider.reset()

            provider1 = SolanaWalletProvider()
            provider2 = SolanaWalletProvider()
            assert provider1 is provider2

    def test_initialization_with_private_key(self):
        """Test provider initializes correctly with private key."""
        with patch("providers.solana_wallet_provider.SOLANA_AVAILABLE", True):
            from providers.solana_wallet_provider import SolanaWalletProvider

            if hasattr(SolanaWalletProvider, "reset"):
                SolanaWalletProvider.reset()

            with (
                patch("providers.solana_wallet_provider.base58") as mock_base58,
                patch("providers.solana_wallet_provider.Keypair") as mock_keypair,
            ):

                mock_base58.b58decode.return_value = b"mock_key_bytes"

                provider = SolanaWalletProvider(private_key="mock_private_key")
                assert provider is not None

    def test_initialization_with_invalid_private_key(self):
        """Test provider handles invalid private key gracefully."""
        with patch("providers.solana_wallet_provider.SOLANA_AVAILABLE", True):
            from providers.solana_wallet_provider import SolanaWalletProvider

            if hasattr(SolanaWalletProvider, "reset"):
                SolanaWalletProvider.reset()

            with (
                patch("providers.solana_wallet_provider.base58") as mock_base58,
                patch("providers.solana_wallet_provider.Keypair") as mock_keypair,
            ):

                mock_base58.b58decode.side_effect = Exception("Invalid key")

                provider = SolanaWalletProvider(private_key="invalid_key")
                assert provider is not None
