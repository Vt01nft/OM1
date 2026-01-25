"""
Solana Wallet Provider for OM1.

This module provides Solana wallet functionality including SOL and USDC transfers.

Part of Bounty #367: OM1 + Smart Assistant + Wallet Payments
"""

import json
import logging
import os
from typing import Any, Dict, Optional

try:
    import base58
    from solana.rpc.api import Client
    from solana.rpc.types import TxOpts
    from solders.keypair import Keypair
    from solders.message import Message
    from solders.pubkey import Pubkey
    from solders.system_program import TransferParams, transfer
    from solders.transaction import Transaction

    SOLANA_AVAILABLE = True
except ImportError:
    SOLANA_AVAILABLE = False
    logging.warning(
        "Solana libraries not available. Install: pip install solana solders base58"
    )

from .singleton import singleton

# USDC Token Mint Addresses
USDC_MINT = {
    "mainnet": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    "devnet": "4zMMC9srt5Ri5X14GAgXhaHii3GnPAEERYPJgZJDncDU",
}

# Wallet storage file
SOL_WALLET_FILE = "sol_wallet.json"


@singleton
class SolanaWalletProvider:
    """
    Solana Wallet Provider for OM1.

    Supports:
    - SOL transfers
    - USDC transfers (SPL Token)
    - Balance checking
    - Transaction status
    - Persistent wallet storage
    """

    NETWORKS = {
        "mainnet": "https://api.mainnet-beta.solana.com",
        "devnet": "https://api.devnet.solana.com",
        "testnet": "https://api.testnet.solana.com",
    }

    EXPLORERS = {
        "mainnet": "https://explorer.solana.com",
        "devnet": "https://explorer.solana.com",
        "testnet": "https://explorer.solana.com",
    }

    def __init__(
        self,
        network: str = "devnet",
        private_key: Optional[str] = None,
    ):
        """
        Initialize the Solana Wallet Provider.

        Parameters
        ----------
        network : str
            Network to connect to ('mainnet', 'devnet', 'testnet')
        private_key : str, optional
            Base58 encoded private key. If None, loads from file or creates new.
        """
        self.network = network.lower()
        self._running = False
        self.client = None
        self.keypair = None
        self.wallet_address = None

        if not SOLANA_AVAILABLE:
            logging.error("Solana libraries not available")
            return

        # Connect to network
        rpc_url = self.NETWORKS.get(self.network, self.NETWORKS["devnet"])
        self.client = Client(rpc_url)
        self.explorer_base = self.EXPLORERS.get(self.network, self.EXPLORERS["devnet"])

        # Setup wallet
        if private_key:
            self._import_wallet(private_key)
        else:
            self._load_or_create_wallet()

        if self.keypair:
            self.wallet_address = str(self.keypair.pubkey())
            self.usdc_mint = USDC_MINT.get(self.network, USDC_MINT["devnet"])
            logging.info(f"Solana Wallet: {self.wallet_address} on {self.network}")

    def _import_wallet(self, private_key: str):
        """Import wallet from base58 private key."""
        try:
            key_bytes = base58.b58decode(private_key)
            self.keypair = Keypair.from_bytes(key_bytes)
            self._save_wallet(private_key)
            logging.info("Imported Solana wallet")
        except Exception as e:
            logging.error(f"Failed to import wallet: {e}")
            self.keypair = None

    def _load_or_create_wallet(self):
        """Load existing wallet or create new one."""
        # Try to load from file
        if os.path.exists(SOL_WALLET_FILE):
            try:
                with open(SOL_WALLET_FILE, "r") as f:
                    data = json.load(f)
                    private_key = data.get("private_key")
                    if private_key:
                        self._import_wallet(private_key)
                        logging.info("Loaded existing Solana wallet")
                        return
            except Exception as e:
                logging.warning(f"Failed to load wallet file: {e}")

        # Create new wallet
        self.keypair = Keypair()
        private_key = base58.b58encode(bytes(self.keypair)).decode()

        self._save_wallet(private_key)

        address = str(self.keypair.pubkey())
        logging.info(f"Created new Solana wallet: {address}")
        print(f"\n{'='*60}")
        print("🔐 NEW SOLANA WALLET CREATED")
        print(f"{'='*60}")
        print(f"   Address: {address}")
        print(f"   Network: {self.network}")
        print(f"   Private Key: {private_key}")
        print("\n   ⚠️  Get free devnet SOL from:")
        print("   https://faucet.solana.com")
        print(f"   Or run: solana airdrop 2 {address} --url devnet")
        print(f"{'='*60}\n")

    def _save_wallet(self, private_key: str):
        """Save wallet to file."""
        try:
            with open(SOL_WALLET_FILE, "w") as f:
                json.dump(
                    {
                        "address": str(self.keypair.pubkey()) if self.keypair else None,
                        "private_key": private_key,
                        "network": self.network,
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logging.error(f"Failed to save wallet: {e}")

    def start(self) -> None:
        """Start the provider."""
        self._running = True
        logging.info("SolanaWalletProvider started")

    def stop(self) -> None:
        """Stop the provider."""
        self._running = False
        logging.info("SolanaWalletProvider stopped")

    def is_connected(self) -> bool:
        """Check if connected to network."""
        if not self.client:
            return False
        try:
            self.client.is_connected()
            return True
        except:
            return False

    def get_balance(self) -> float:
        """Get SOL balance."""
        if not self.client or not self.keypair:
            return 0.0
        try:
            response = self.client.get_balance(self.keypair.pubkey())
            lamports = response.value
            return lamports / 1_000_000_000  # Convert lamports to SOL
        except Exception as e:
            logging.error(f"Failed to get SOL balance: {e}")
            return 0.0

    def get_usdc_balance(self) -> float:
        """Get USDC balance (SPL Token)."""
        if not self.client or not self.keypair:
            return 0.0
        try:
            from solana.rpc.types import TokenAccountOpts

            mint = Pubkey.from_string(self.usdc_mint)

            # Use TokenAccountOpts for proper filtering
            opts = TokenAccountOpts(mint=mint)
            response = self.client.get_token_accounts_by_owner_json_parsed(
                self.keypair.pubkey(), opts
            )

            if response.value:
                for account in response.value:
                    try:
                        parsed_data = account.account.data.parsed
                        if isinstance(parsed_data, dict):
                            info = parsed_data.get("info", {})
                            token_amount = info.get("tokenAmount", {})
                            ui_amount = token_amount.get("uiAmount")
                            if ui_amount is not None:
                                return float(ui_amount)
                    except (AttributeError, KeyError, TypeError):
                        continue
            return 0.0
        except Exception as e:
            logging.error(f"Failed to get USDC balance: {e}")
            return 0.0

    def send_sol(self, to_address: str, amount: float) -> Dict[str, Any]:
        """
        Send SOL to an address.

        Parameters
        ----------
        to_address : str
            Recipient Solana address
        amount : float
            Amount in SOL

        Returns
        -------
        dict
            Transaction result
        """
        if not self.client or not self.keypair:
            return {"success": False, "error": "Wallet not initialized"}

        try:
            # Validate and parse address
            to_pubkey = Pubkey.from_string(to_address)
            lamports = int(amount * 1_000_000_000)

            # Check balance
            balance = self.get_balance()
            if balance < amount:
                return {
                    "success": False,
                    "error": f"Insufficient SOL. Have: {balance:.4f}, Need: {amount:.4f}",
                }

            # Create transfer instruction
            transfer_ix = transfer(
                TransferParams(
                    from_pubkey=self.keypair.pubkey(),
                    to_pubkey=to_pubkey,
                    lamports=lamports,
                )
            )

            # Get recent blockhash
            recent_blockhash = self.client.get_latest_blockhash().value.blockhash

            # Create and sign transaction
            msg = Message.new_with_blockhash(
                [transfer_ix], self.keypair.pubkey(), recent_blockhash
            )
            tx = Transaction.new_unsigned(msg)
            tx.sign([self.keypair], recent_blockhash)

            # Send transaction
            opts = TxOpts(skip_preflight=False, preflight_commitment="confirmed")
            result = self.client.send_transaction(tx, opts=opts)

            signature = str(result.value)
            cluster_param = (
                f"?cluster={self.network}" if self.network != "mainnet" else ""
            )
            explorer_url = f"{self.explorer_base}/tx/{signature}{cluster_param}"

            logging.info(f"SOL sent: {amount} to {to_address}")
            logging.info(f"TX: {signature}")

            return {
                "success": True,
                "signature": signature,
                "tx_hash": signature,  # Alias for compatibility
                "amount": amount,
                "token": "SOL",
                "recipient": to_address,
                "network": self.network,
                "explorer_url": explorer_url,
            }

        except Exception as e:
            logging.error(f"SOL transaction failed: {e}")
            return {"success": False, "error": str(e)}

    def send_usdc(self, to_address: str, amount: float) -> Dict[str, Any]:
        """
        Send USDC to an address.

        Note: Full SPL token transfers require associated token accounts.
        This is a simplified implementation.

        Parameters
        ----------
        to_address : str
            Recipient address
        amount : float
            Amount in USDC

        Returns
        -------
        dict
            Transaction result
        """
        if not self.client or not self.keypair:
            return {"success": False, "error": "Wallet not initialized"}

        try:
            # Check USDC balance
            usdc_balance = self.get_usdc_balance()
            if usdc_balance < amount:
                return {
                    "success": False,
                    "error": f"Insufficient USDC. Have: {usdc_balance:.2f}, Need: {amount:.2f}",
                }

            # For full SPL token transfer, we need spl-token library
            # This requires associated token accounts setup
            # For now, return informative error for production use

            # Check if spl-token is available
            try:
                from spl.token.constants import TOKEN_PROGRAM_ID
                from spl.token.instructions import (
                    TransferCheckedParams,
                    transfer_checked,
                )

                # TODO: Implement full SPL token transfer
                # This requires:
                # 1. Get sender's associated token account
                # 2. Get/create recipient's associated token account
                # 3. Build transfer_checked instruction
                # 4. Sign and send transaction

                return {
                    "success": False,
                    "error": "USDC transfer requires SPL token setup. Use SOL for now.",
                }
            except ImportError:
                return {
                    "success": False,
                    "error": "SPL token library not installed. Run: pip install spl-token",
                }

        except Exception as e:
            logging.error(f"USDC transfer failed: {e}")
            return {"success": False, "error": str(e)}

    def send(
        self, to_address: str, amount: float, token: str = "SOL"
    ) -> Dict[str, Any]:
        """
        Universal send method.

        Parameters
        ----------
        to_address : str
            Recipient address
        amount : float
            Amount to send
        token : str
            Token to send ('SOL' or 'USDC')
        """
        token = token.upper()
        if token == "SOL":
            return self.send_sol(to_address, amount)
        elif token == "USDC":
            return self.send_usdc(to_address, amount)
        else:
            return {"success": False, "error": f"Unknown token: {token}"}

    def get_transaction_status(self, signature: str) -> Dict[str, Any]:
        """Get transaction status."""
        if not self.client:
            return {"status": "unknown", "error": "Not connected"}

        try:
            from solders.signature import Signature

            sig = Signature.from_string(signature)
            response = self.client.get_signature_statuses([sig])

            if response.value and response.value[0]:
                status = response.value[0]
                if status.err:
                    return {"status": "failed", "error": str(status.err)}
                if status.confirmation_status:
                    return {
                        "status": "successful",
                        "confirmations": status.confirmations,
                        "slot": status.slot,
                    }
            return {"status": "pending"}
        except Exception as e:
            return {"status": "unknown", "error": str(e)}

    def request_airdrop(self, amount: float = 1.0) -> Dict[str, Any]:
        """Request airdrop (devnet/testnet only)."""
        if self.network == "mainnet":
            return {"success": False, "error": "Airdrop not available on mainnet"}

        if not self.client or not self.keypair:
            return {"success": False, "error": "Wallet not initialized"}

        try:
            lamports = int(amount * 1_000_000_000)
            response = self.client.request_airdrop(self.keypair.pubkey(), lamports)
            signature = str(response.value)

            return {
                "success": True,
                "signature": signature,
                "amount": amount,
                "message": f"Requested {amount} SOL airdrop",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_wallet_info(self) -> Dict[str, Any]:
        """Get wallet information."""
        cluster_param = f"?cluster={self.network}" if self.network != "mainnet" else ""
        return {
            "address": self.wallet_address,
            "network": self.network,
            "connected": self.is_connected(),
            "sol_balance": self.get_balance(),
            "usdc_balance": self.get_usdc_balance(),
            "explorer": f"{self.explorer_base}/address/{self.wallet_address}{cluster_param}",
        }

    def get_wallet_summary(self) -> str:
        """Get formatted wallet summary."""
        sol = self.get_balance()
        usdc = self.get_usdc_balance()
        short_addr = f"{self.wallet_address[:8]}...{self.wallet_address[-4:]}"
        return f"Solana wallet ({short_addr}): {sol:.4f} SOL, {usdc:.2f} USDC"

    def format_transaction_result(self, result: Dict[str, Any]) -> str:
        """Format transaction result for display."""
        if result["success"]:
            return (
                f"✅ Sent {result['amount']} {result['token']} to {result['recipient'][:12]}...\n"
                f"   TX: {result['signature']}\n"
                f"   Explorer: {result['explorer_url']}"
            )
        else:
            return f"❌ Failed: {result.get('error', 'Unknown error')}"
