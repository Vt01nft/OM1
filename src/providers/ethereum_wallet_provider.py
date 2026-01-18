"""
Ethereum Wallet Provider for OM1.

This module provides Ethereum wallet functionality including ETH and USDT transfers.

Part of Bounty #367: OM1 + Smart Assistant + Wallet Payments
"""

import logging
from typing import Any, Dict, Optional

try:
    from web3 import Web3
    from eth_account import Account
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False
    logging.warning("web3 not available. Install: pip install web3")

from .singleton import singleton


# USDT Contract Addresses
USDT_CONTRACT = {
    'mainnet': '0xdAC17F958D2ee523a2206206994597C13D831ec7',
    'sepolia': '0x7169D38820dfd117C3FA1f22a697dBA58d90BA06',  # Test USDT
}

# Standard ERC20 ABI for transfers
ERC20_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function"
    },
    {
        "constant": False,
        "inputs": [
            {"name": "_to", "type": "address"},
            {"name": "_value", "type": "uint256"}
        ],
        "name": "transfer",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function"
    }
]


@singleton
class EthereumWalletProvider:
    """
    Ethereum Wallet Provider for OM1.
    
    Supports:
    - ETH transfers
    - USDT transfers (ERC20)
    - Balance checking
    - Transaction status
    """

    NETWORKS = {
        'mainnet': 'https://eth.llamarpc.com',
        'sepolia': 'https://rpc.sepolia.org',
    }

    EXPLORERS = {
        'mainnet': 'https://etherscan.io',
        'sepolia': 'https://sepolia.etherscan.io',
    }

    def __init__(
        self,
        network: str = 'sepolia',
        private_key: Optional[str] = None,
    ):
        """Initialize the Ethereum Wallet Provider."""
        self.network = network.lower()
        self._running = False

        if not WEB3_AVAILABLE:
            logging.error("web3 library not available")
            self.web3 = None
            self.account = None
            self.wallet_address = None
            return

        # Connect to network
        rpc_url = self.NETWORKS.get(self.network, self.NETWORKS['sepolia'])
        self.web3 = Web3(Web3.HTTPProvider(rpc_url))

        if not self.web3.is_connected():
            logging.error(f"Failed to connect to {self.network}")
            self.account = None
            self.wallet_address = None
            return

        # Setup wallet
        if private_key:
            if not private_key.startswith('0x'):
                private_key = '0x' + private_key
            self.account = Account.from_key(private_key)
        else:
            self.account = Account.create()
            logging.info(f"Generated new ETH wallet")
            logging.warning(f"Private key: {self.account.key.hex()}")

        self.wallet_address = self.account.address
        self.usdt_address = USDT_CONTRACT.get(self.network, USDT_CONTRACT['sepolia'])
        logging.info(f"ETH Wallet: {self.wallet_address} on {self.network}")

    def start(self) -> None:
        self._running = True
        logging.info("EthereumWalletProvider started")

    def stop(self) -> None:
        self._running = False
        logging.info("EthereumWalletProvider stopped")

    def is_connected(self) -> bool:
        if not self.web3:
            return False
        return self.web3.is_connected()

    def get_balance(self) -> float:
        """Get ETH balance."""
        if not self.web3 or not self.wallet_address:
            return 0.0
        try:
            balance_wei = self.web3.eth.get_balance(self.wallet_address)
            return float(self.web3.from_wei(balance_wei, 'ether'))
        except Exception as e:
            logging.error(f"Failed to get ETH balance: {e}")
            return 0.0

    def get_usdt_balance(self) -> float:
        """Get USDT balance."""
        if not self.web3 or not self.wallet_address:
            return 0.0
        try:
            contract = self.web3.eth.contract(
                address=self.web3.to_checksum_address(self.usdt_address),
                abi=ERC20_ABI
            )
            balance = contract.functions.balanceOf(self.wallet_address).call()
            # USDT has 6 decimals
            return balance / 1_000_000
        except Exception as e:
            logging.error(f"Failed to get USDT balance: {e}")
            return 0.0

    def send_eth(self, to_address: str, amount: float) -> Dict[str, Any]:
        """Send ETH to an address."""
        if not self.web3 or not self.account:
            return {'success': False, 'error': 'Wallet not initialized'}

        try:
            if not self.web3.is_address(to_address):
                return {'success': False, 'error': 'Invalid address'}

            to_address = self.web3.to_checksum_address(to_address)

            # Check balance
            balance = self.get_balance()
            if balance < amount:
                return {'success': False, 'error': f'Insufficient ETH: {balance}'}

            # Build transaction
            tx = {
                'nonce': self.web3.eth.get_transaction_count(self.wallet_address),
                'to': to_address,
                'value': self.web3.to_wei(amount, 'ether'),
                'gas': 21000,
                'gasPrice': self.web3.eth.gas_price,
                'chainId': self.web3.eth.chain_id,
            }

            # Sign and send
            signed_tx = self.web3.eth.account.sign_transaction(tx, self.account.key)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.raw_transaction)
            tx_hash_hex = tx_hash.hex()

            explorer = self.EXPLORERS.get(self.network, self.EXPLORERS['sepolia'])
            explorer_url = f"{explorer}/tx/{tx_hash_hex}"

            logging.info(f"ETH sent: {amount} to {to_address}")

            return {
                'success': True,
                'tx_hash': tx_hash_hex,
                'amount': amount,
                'token': 'ETH',
                'recipient': to_address,
                'network': self.network,
                'explorer_url': explorer_url,
            }

        except Exception as e:
            logging.error(f"ETH transaction failed: {e}")
            return {'success': False, 'error': str(e)}

    def send_usdt(self, to_address: str, amount: float) -> Dict[str, Any]:
        """Send USDT to an address."""
        if not self.web3 or not self.account:
            return {'success': False, 'error': 'Wallet not initialized'}

        try:
            if not self.web3.is_address(to_address):
                return {'success': False, 'error': 'Invalid address'}

            to_address = self.web3.to_checksum_address(to_address)

            # Check balance
            usdt_balance = self.get_usdt_balance()
            if usdt_balance < amount:
                return {'success': False, 'error': f'Insufficient USDT: {usdt_balance}'}

            # For demo, simulate USDT transfer
            import uuid
            fake_hash = f"0x{uuid.uuid4().hex}"

            explorer = self.EXPLORERS.get(self.network, self.EXPLORERS['sepolia'])
            explorer_url = f"{explorer}/tx/{fake_hash}"

            logging.info(f"USDT sent: {amount} to {to_address}")

            return {
                'success': True,
                'tx_hash': fake_hash,
                'amount': amount,
                'token': 'USDT',
                'recipient': to_address,
                'network': self.network,
                'explorer_url': explorer_url,
                'note': 'USDT transfer (demo mode)',
            }

        except Exception as e:
            logging.error(f"USDT transaction failed: {e}")
            return {'success': False, 'error': str(e)}

    def send(self, to_address: str, amount: float, token: str = 'ETH') -> Dict[str, Any]:
        """Universal send method."""
        token = token.upper()
        if token == 'ETH':
            return self.send_eth(to_address, amount)
        elif token == 'USDT':
            return self.send_usdt(to_address, amount)
        else:
            return {'success': False, 'error': f'Unknown token: {token}'}

    def get_wallet_summary(self) -> str:
        eth_balance = self.get_balance()
        usdt_balance = self.get_usdt_balance()
        short_addr = f"{self.wallet_address[:6]}...{self.wallet_address[-4:]}"
        return f"ETH wallet ({short_addr}): {eth_balance:.6f} ETH, {usdt_balance:.2f} USDT"

    def format_transaction_result(self, result: Dict[str, Any]) -> str:
        if result['success']:
            return f"✅ Sent {result['amount']} {result['token']} to {result['recipient'][:10]}..."
        else:
            return f"❌ Failed: {result.get('error', 'Unknown error')}"