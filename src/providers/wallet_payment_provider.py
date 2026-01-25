"""Wallet Payment Provider for OM1.

Bounty #367: OM1 + Smart Assistant + Wallet Payments

This provider provides a unified interface for multi-chain wallet payments,
supporting both Solana and Ethereum networks with automatic token detection
and transaction management.

Supported Networks:
- Solana (devnet/mainnet) - SOL, USDC
- Ethereum (sepolia/mainnet) - ETH, USDT

Features:
- Multi-chain wallet management
- Automatic price conversion (USD to crypto)
- Transaction history tracking
- Contact management
- Balance checking across all tokens
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from providers.singleton import singleton
except ImportError:

    def singleton(cls):
        """Singleton decorator fallback."""
        return cls


logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# PRICE SERVICE
# ═══════════════════════════════════════════════════════════════════


class PriceService:
    """Fetches live cryptocurrency prices from CoinGecko API."""

    def __init__(self):
        """Initialize the price service."""
        self.sol_price = 140.0
        self.eth_price = 3500.0
        self._fetch_prices()

    def _fetch_prices(self) -> None:
        """Fetch current prices from CoinGecko."""
        try:
            import urllib.request

            url = "https://api.coingecko.com/api/v3/simple/price?ids=solana,ethereum&vs_currencies=usd"
            req = urllib.request.Request(url, headers={"User-Agent": "OM1/1.0"})
            with urllib.request.urlopen(req, timeout=5) as r:
                data = json.loads(r.read().decode())
                self.sol_price = data.get("solana", {}).get("usd", 140.0)
                self.eth_price = data.get("ethereum", {}).get("usd", 3500.0)
                logger.info(f"Prices: SOL=${self.sol_price}, ETH=${self.eth_price}")
        except Exception as e:
            logger.warning(f"Failed to fetch prices: {e}")

    def usd_to_sol(self, usd: float) -> float:
        """Convert USD to SOL."""
        return usd / self.sol_price

    def usd_to_eth(self, usd: float) -> float:
        """Convert USD to ETH."""
        return usd / self.eth_price

    def sol_to_usd(self, sol: float) -> float:
        """Convert SOL to USD."""
        return sol * self.sol_price

    def eth_to_usd(self, eth: float) -> float:
        """Convert ETH to USD."""
        return eth * self.eth_price


# ═══════════════════════════════════════════════════════════════════
# WALLET PAYMENT PROVIDER
# ═══════════════════════════════════════════════════════════════════


@singleton
class WalletPaymentProvider:
    """Unified multi-chain wallet payment provider.

    Provides a single interface for managing wallets across Solana and Ethereum,
    processing payments, and tracking transactions.

    Example:
        wallet = WalletPaymentProvider()
        balances = wallet.get_all_balances()
        result = wallet.pay(token='SOL', amount=0.1, recipient='store_address')
    """

    # Supported tokens and their chains
    TOKEN_CHAINS = {
        "SOL": "solana",
        "USDC": "solana",
        "ETH": "ethereum",
        "USDT": "ethereum",
    }

    # Default store addresses for demo
    STORE_ADDRESSES = {
        "SOL": "Hxro1u6mfWC4ZJJdxCgkEfGdJoc3nnsAw4Jebast1SX8",
        "USDC": "Hxro1u6mfWC4ZJJdxCgkEfGdJoc3nnsAw4Jebast1SX8",
        "ETH": "0x742d35Cc6634C0532925a3b844Bc9e7595f5bE91",
        "USDT": "0x742d35Cc6634C0532925a3b844Bc9e7595f5bE91",
    }

    def __init__(
        self,
        solana_network: str = "devnet",
        ethereum_network: str = "sepolia",
        solana_private_key: Optional[str] = None,
        ethereum_private_key: Optional[str] = None,
        history_file: str = "wallet_history.json",
        contacts_file: str = "wallet_contacts.json",
    ):
        """Initialize the Wallet Payment Provider.

        Args:
            solana_network: Solana network ('devnet' or 'mainnet-beta').
            ethereum_network: Ethereum network ('sepolia' or 'mainnet').
            solana_private_key: Optional Solana private key.
            ethereum_private_key: Optional Ethereum private key.
            history_file: Path to transaction history file.
            contacts_file: Path to contacts file.
        """
        self.solana_network = solana_network
        self.ethereum_network = ethereum_network
        self.history_file = history_file
        self.contacts_file = contacts_file

        # Price service
        self.price = PriceService()

        # Wallet instances
        self.solana_wallet = None
        self.ethereum_wallet = None

        # Initialize Solana wallet
        self._init_solana(solana_network, solana_private_key)

        # Initialize Ethereum wallet
        self._init_ethereum(ethereum_network, ethereum_private_key)

        # Load data
        self.history = self._load_json(history_file, [])
        self.contacts = self._load_json(contacts_file, {})

        logger.info("WalletPaymentProvider initialized")

    def _init_solana(self, network: str, private_key: Optional[str]) -> None:
        """Initialize Solana wallet."""
        try:
            from providers.solana_wallet_provider import SolanaWalletProvider

            self.solana_wallet = SolanaWalletProvider(
                network=network, private_key=private_key
            )
            logger.info(f"Solana wallet: {self.solana_wallet.wallet_address}")
        except ImportError:
            logger.warning("Solana wallet provider not available")
        except Exception as e:
            logger.error(f"Solana wallet init failed: {e}")

    def _init_ethereum(self, network: str, private_key: Optional[str]) -> None:
        """Initialize Ethereum wallet."""
        try:
            from providers.ethereum_wallet_provider import EthereumWalletProvider

            self.ethereum_wallet = EthereumWalletProvider(
                network=network, private_key=private_key
            )
            if self.ethereum_wallet and self.ethereum_wallet.wallet_address:
                logger.info(f"Ethereum wallet: {self.ethereum_wallet.wallet_address}")
        except ImportError:
            logger.warning("Ethereum wallet provider not available")
        except Exception as e:
            logger.error(f"Ethereum wallet init failed: {e}")

    def _load_json(self, path: str, default: Any) -> Any:
        """Load JSON file."""
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
        return default

    def _save_json(self, path: str, data: Any) -> bool:
        """Save JSON file."""
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save {path}: {e}")
            return False

    # ─────────────────────────────────────────────────────────────
    # Balance Methods
    # ─────────────────────────────────────────────────────────────

    def get_balance(self, token: str) -> float:
        """Get balance for a specific token.

        Args:
            token: Token symbol (SOL, ETH, USDC, USDT).

        Returns
        -------
            Token balance.
        """
        token = token.upper()

        if token == "SOL" and self.solana_wallet:
            return self.solana_wallet.get_balance()
        elif token == "USDC" and self.solana_wallet:
            return self.solana_wallet.get_usdc_balance()
        elif token == "ETH" and self.ethereum_wallet:
            return self.ethereum_wallet.get_balance()
        elif token == "USDT" and self.ethereum_wallet:
            return self.ethereum_wallet.get_usdt_balance()

        return 0.0

    def get_all_balances(self) -> Dict[str, float]:
        """Get balances for all supported tokens.

        Returns
        -------
            Dictionary of token -> balance.
        """
        return {
            "SOL": self.get_balance("SOL"),
            "USDC": self.get_balance("USDC"),
            "ETH": self.get_balance("ETH"),
            "USDT": self.get_balance("USDT"),
        }

    def get_balances_with_usd(self) -> Dict[str, Dict[str, float]]:
        """Get balances with USD values.

        Returns
        -------
            Dictionary with balance and USD value for each token.
        """
        balances = self.get_all_balances()
        return {
            "SOL": {
                "balance": balances["SOL"],
                "usd": self.price.sol_to_usd(balances["SOL"]),
            },
            "USDC": {"balance": balances["USDC"], "usd": balances["USDC"]},
            "ETH": {
                "balance": balances["ETH"],
                "usd": self.price.eth_to_usd(balances["ETH"]),
            },
            "USDT": {"balance": balances["USDT"], "usd": balances["USDT"]},
        }

    # ─────────────────────────────────────────────────────────────
    # Wallet Info
    # ─────────────────────────────────────────────────────────────

    def get_wallet_info(self, chain: str) -> Optional[Dict[str, Any]]:
        """Get wallet information for a chain.

        Args:
            chain: 'solana' or 'ethereum'.

        Returns
        -------
            Wallet info dict or None.
        """
        if chain == "solana" and self.solana_wallet:
            return {
                "chain": "solana",
                "network": self.solana_network,
                "address": self.solana_wallet.wallet_address,
                "balance": self.get_balance("SOL"),
                "tokens": ["SOL", "USDC"],
                "faucet": "https://faucet.solana.com",
            }
        elif chain == "ethereum" and self.ethereum_wallet:
            return {
                "chain": "ethereum",
                "network": self.ethereum_network,
                "address": self.ethereum_wallet.wallet_address,
                "balance": self.get_balance("ETH"),
                "tokens": ["ETH", "USDT"],
                "faucet": "https://sepoliafaucet.com",
            }
        return None

    def get_all_wallets(self) -> List[Dict[str, Any]]:
        """Get info for all connected wallets."""
        wallets = []

        sol_info = self.get_wallet_info("solana")
        if sol_info:
            wallets.append(sol_info)

        eth_info = self.get_wallet_info("ethereum")
        if eth_info:
            wallets.append(eth_info)

        return wallets

    # ─────────────────────────────────────────────────────────────
    # Payment Methods
    # ─────────────────────────────────────────────────────────────

    def calculate_payment(self, usd_amount: float) -> Dict[str, float]:
        """Calculate payment amounts in all tokens.

        Args:
            usd_amount: Amount in USD.

        Returns
        -------
            Dictionary of token -> amount.
        """
        return {
            "SOL": self.price.usd_to_sol(usd_amount),
            "ETH": self.price.usd_to_eth(usd_amount),
            "USDC": usd_amount,
            "USDT": usd_amount,
        }

    def can_afford(self, token: str, amount: float) -> bool:
        """Check if wallet can afford a payment.

        Args:
            token: Token to pay with.
            amount: Amount required.

        Returns
        -------
            True if sufficient balance.
        """
        balance = self.get_balance(token)
        return balance >= amount

    def find_affordable_token(self, amounts: Dict[str, float]) -> Optional[str]:
        """Find a token with sufficient balance.

        Args:
            amounts: Dictionary of token -> required amount.

        Returns
        -------
            Token name or None if none affordable.
        """
        balances = self.get_all_balances()

        for token in ["SOL", "ETH", "USDC", "USDT"]:
            if balances.get(token, 0) >= amounts.get(token, float("inf")):
                return token

        return None

    def pay(
        self,
        token: str,
        amount: float,
        recipient: str,
        description: str = "",
    ) -> Dict[str, Any]:
        """Process a payment.

        Args:
            token: Token to send (SOL, ETH, USDC, USDT).
            amount: Amount to send.
            recipient: Recipient address.
            description: Transaction description.

        Returns
        -------
            Transaction result dict with 'success', 'tx_hash'/'signature', 'error'.
        """
        token = token.upper()
        chain = self.TOKEN_CHAINS.get(token)

        # Validate
        if not chain:
            return {"success": False, "error": f"Unknown token: {token}"}

        if not self.can_afford(token, amount):
            balance = self.get_balance(token)
            return {
                "success": False,
                "error": f"Insufficient {token}: have {balance:.4f}, need {amount:.4f}",
            }

        # Send
        result = self._send(token, amount, recipient)

        # Record transaction
        if result.get("success"):
            tx = {
                "timestamp": datetime.now().isoformat(),
                "type": "payment",
                "token": token,
                "amount": amount,
                "recipient": recipient,
                "description": description,
                "tx_hash": result.get("signature") or result.get("tx_hash"),
                "status": "successful",
            }
            self.history.append(tx)
            self._save_json(self.history_file, self.history)
            logger.info(f"Payment successful: {amount} {token} to {recipient[:20]}...")

        return result

    def _send(self, token: str, amount: float, recipient: str) -> Dict[str, Any]:
        """Internal send method."""
        if token in ["SOL", "USDC"]:
            if self.solana_wallet:
                return self.solana_wallet.send(recipient, amount, token)
            return {"success": False, "error": "Solana wallet not connected"}

        elif token in ["ETH", "USDT"]:
            if self.ethereum_wallet:
                return self.ethereum_wallet.send(recipient, amount, token)
            return {"success": False, "error": "Ethereum wallet not connected"}

        return {"success": False, "error": f"Unknown token: {token}"}

    def pay_store(
        self,
        token: str,
        amount: float,
        item: str,
    ) -> Dict[str, Any]:
        """Pay to store address.

        Args:
            token: Token to pay with.
            amount: Amount to pay.
            item: Item being purchased.

        Returns
        -------
            Transaction result.
        """
        store_address = self.STORE_ADDRESSES.get(token.upper())
        if not store_address:
            return {"success": False, "error": f"No store address for {token}"}

        result = self.pay(
            token=token,
            amount=amount,
            recipient=store_address,
            description=f"Order: {item}",
        )

        # Update history with order info
        if result.get("success") and self.history:
            self.history[-1]["type"] = "order"
            self.history[-1]["item"] = item
            self._save_json(self.history_file, self.history)

        return result

    # ─────────────────────────────────────────────────────────────
    # Contact Management
    # ─────────────────────────────────────────────────────────────

    def add_contact(self, name: str, address: str, token: str = "SOL") -> bool:
        """Add a contact.

        Args:
            name: Contact name.
            address: Wallet address.
            token: Default token for this contact.

        Returns
        -------
            True if added successfully.
        """
        self.contacts[name.lower()] = {
            "name": name,
            "address": address,
            "token": token.upper(),
            "added": datetime.now().isoformat(),
        }
        return self._save_json(self.contacts_file, self.contacts)

    def remove_contact(self, name: str) -> bool:
        """Remove a contact."""
        if name.lower() in self.contacts:
            del self.contacts[name.lower()]
            return self._save_json(self.contacts_file, self.contacts)
        return False

    def get_contact(self, name: str) -> Optional[Dict[str, Any]]:
        """Get contact by name."""
        return self.contacts.get(name.lower())

    def get_all_contacts(self) -> Dict[str, Dict[str, Any]]:
        """Get all contacts."""
        return self.contacts

    def pay_contact(
        self,
        contact_name: str,
        amount: float,
        token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Pay a contact.

        Args:
            contact_name: Contact name.
            amount: Amount to send.
            token: Token to use (defaults to contact's preferred token).

        Returns
        -------
            Transaction result.
        """
        contact = self.get_contact(contact_name)
        if not contact:
            return {"success": False, "error": f"Contact not found: {contact_name}"}

        token = token or contact.get("token", "SOL")

        result = self.pay(
            token=token,
            amount=amount,
            recipient=contact["address"],
            description=f"Transfer to {contact['name']}",
        )

        # Update history
        if result.get("success") and self.history:
            self.history[-1]["type"] = "transfer"
            self.history[-1]["to"] = contact["name"]
            self._save_json(self.history_file, self.history)

        return result

    # ─────────────────────────────────────────────────────────────
    # Transaction History
    # ─────────────────────────────────────────────────────────────

    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent transaction history."""
        return self.history[-limit:] if self.history else []

    def get_orders(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent orders."""
        orders = [tx for tx in self.history if tx.get("type") == "order"]
        return orders[-limit:] if orders else []

    def get_transfers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent transfers."""
        transfers = [tx for tx in self.history if tx.get("type") == "transfer"]
        return transfers[-limit:] if transfers else []

    # ─────────────────────────────────────────────────────────────
    # Status
    # ─────────────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Get provider status."""
        return {
            "solana_connected": self.solana_wallet is not None,
            "ethereum_connected": self.ethereum_wallet is not None,
            "solana_address": (
                self.solana_wallet.wallet_address if self.solana_wallet else None
            ),
            "ethereum_address": (
                self.ethereum_wallet.wallet_address if self.ethereum_wallet else None
            ),
            "prices": {
                "SOL": self.price.sol_price,
                "ETH": self.price.eth_price,
            },
            "total_transactions": len(self.history),
            "total_contacts": len(self.contacts),
        }

    def __repr__(self) -> str:
        """Return string representation."""
        return f"WalletPaymentProvider(sol={self.solana_wallet is not None}, eth={self.ethereum_wallet is not None})"
