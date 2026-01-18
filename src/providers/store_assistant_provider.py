"""
Store Assistant Provider for OM1.

Simulates a smart assistant (like Home Assistant) for ordering products
with crypto payments.

Part of Bounty #367: OM1 + Smart Assistant + Wallet Payments
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .singleton import singleton


@singleton
class StoreAssistantProvider:
    """
    Smart Store Assistant for OM1.
    
    Handles:
    - Product catalog
    - Order creation
    - Payment processing coordination
    - Order status tracking
    """

    # Product catalog with USD prices
    PRODUCTS = {
        'coffee': {'name': 'Coffee', 'price_usd': 5.00, 'category': 'drinks'},
        'tea': {'name': 'Tea', 'price_usd': 3.50, 'category': 'drinks'},
        'pizza': {'name': 'Pizza', 'price_usd': 15.00, 'category': 'food'},
        'burger': {'name': 'Burger', 'price_usd': 10.00, 'category': 'food'},
        'salad': {'name': 'Salad', 'price_usd': 8.00, 'category': 'food'},
        'sandwich': {'name': 'Sandwich', 'price_usd': 7.00, 'category': 'food'},
        'soda': {'name': 'Soda', 'price_usd': 2.50, 'category': 'drinks'},
        'water': {'name': 'Water', 'price_usd': 1.50, 'category': 'drinks'},
        'fries': {'name': 'Fries', 'price_usd': 4.00, 'category': 'food'},
        'ice cream': {'name': 'Ice Cream', 'price_usd': 5.00, 'category': 'dessert'},
    }

    # Store wallet addresses (merchant receives payments)
    STORE_WALLETS = {
        'SOL': 'Hxro1u6mfWC4ZJJdxCgkEfGdJoc3nnsAw4Jebast1SX8',
        'USDC': 'Hxro1u6mfWC4ZJJdxCgkEfGdJoc3nnsAw4Jebast1SX8',
        'ETH': '0x742d35Cc6634C0532925a3b844Bc9e7595f5bE91',
        'USDT': '0x742d35Cc6634C0532925a3b844Bc9e7595f5bE91',
    }

    def __init__(self):
        """Initialize the Store Assistant."""
        self.orders: Dict[str, Dict] = {}
        self._running = False
        logging.info("StoreAssistantProvider initialized")

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def get_product(self, name: str) -> Optional[Dict]:
        """Get product by name."""
        return self.PRODUCTS.get(name.lower())

    def list_products(self, category: Optional[str] = None) -> List[Dict]:
        """List all products, optionally filtered by category."""
        products = []
        for key, prod in self.PRODUCTS.items():
            if category is None or prod['category'] == category.lower():
                products.append({**prod, 'id': key})
        return products

    def calculate_crypto_price(self, usd_price: float, token: str, sol_price: float = 140.0, eth_price: float = 3500.0) -> float:
        """
        Convert USD price to crypto amount.
        
        Parameters
        ----------
        usd_price : float
            Price in USD
        token : str
            Token type (SOL, USDC, ETH, USDT)
        sol_price : float
            Current SOL/USD price
        eth_price : float
            Current ETH/USD price
        """
        token = token.upper()
        if token == 'SOL':
            return round(usd_price / sol_price, 6)
        elif token == 'ETH':
            return round(usd_price / eth_price, 6)
        elif token in ['USDC', 'USDT']:
            return usd_price  # Stablecoins are 1:1 with USD
        else:
            return usd_price

    def create_order(self, product_name: str, quantity: int = 1) -> Dict[str, Any]:
        """
        Create a new order.
        
        Returns order details with prices in all supported tokens.
        """
        product = self.get_product(product_name)
        
        if not product:
            return {
                'success': False,
                'error': f"Product '{product_name}' not found",
                'available': list(self.PRODUCTS.keys()),
            }

        order_id = str(uuid.uuid4())[:8].upper()
        total_usd = product['price_usd'] * quantity

        order = {
            'order_id': order_id,
            'product': product['name'],
            'product_id': product_name.lower(),
            'quantity': quantity,
            'price_usd': total_usd,
            'prices': {
                'SOL': self.calculate_crypto_price(total_usd, 'SOL'),
                'USDC': self.calculate_crypto_price(total_usd, 'USDC'),
                'ETH': self.calculate_crypto_price(total_usd, 'ETH'),
                'USDT': self.calculate_crypto_price(total_usd, 'USDT'),
            },
            'status': 'awaiting_payment',
            'created_at': datetime.now().isoformat(),
            'payment_token': None,
            'payment_tx': None,
        }

        self.orders[order_id] = order

        return {
            'success': True,
            'order': order,
        }

    def get_payment_address(self, token: str) -> str:
        """Get store wallet address for the specified token."""
        return self.STORE_WALLETS.get(token.upper(), self.STORE_WALLETS['SOL'])

    def confirm_payment(self, order_id: str, token: str, tx_signature: str) -> Dict[str, Any]:
        """Confirm payment for an order."""
        if order_id not in self.orders:
            return {'success': False, 'error': 'Order not found'}

        order = self.orders[order_id]
        order['status'] = 'paid'
        order['payment_token'] = token.upper()
        order['payment_tx'] = tx_signature
        order['paid_at'] = datetime.now().isoformat()

        return {
            'success': True,
            'order': order,
            'message': f"Payment confirmed! Order #{order_id} is being prepared.",
        }

    def complete_order(self, order_id: str) -> Dict[str, Any]:
        """Mark order as completed."""
        if order_id not in self.orders:
            return {'success': False, 'error': 'Order not found'}

        order = self.orders[order_id]
        order['status'] = 'completed'
        order['completed_at'] = datetime.now().isoformat()

        return {
            'success': True,
            'order': order,
            'message': f"Order #{order_id} complete! Enjoy your {order['product']}!",
        }

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order."""
        if order_id not in self.orders:
            return {'success': False, 'error': 'Order not found'}

        order = self.orders[order_id]
        if order['status'] in ['paid', 'completed']:
            return {'success': False, 'error': 'Cannot cancel paid/completed order'}

        order['status'] = 'cancelled'
        order['cancelled_at'] = datetime.now().isoformat()

        return {'success': True, 'message': f"Order #{order_id} cancelled."}

    def get_order(self, order_id: str) -> Optional[Dict]:
        """Get order by ID."""
        return self.orders.get(order_id)

    def get_order_history(self) -> List[Dict]:
        """Get all orders."""
        return list(self.orders.values())

    def format_product_list(self) -> str:
        """Format products for display."""
        lines = ["🛒 Available Products:"]
        for key, prod in self.PRODUCTS.items():
            lines.append(f"   • {prod['name']}: ${prod['price_usd']:.2f}")
        return "\n".join(lines)

    def format_order_options(self, order: Dict) -> str:
        """Format payment options for display."""
        prices = order['prices']
        return (
            f"💰 Pay ${order['price_usd']:.2f} with:\n"
            f"   • SOL ({prices['SOL']:.4f} SOL)\n"
            f"   • USDC ({prices['USDC']:.2f} USDC)\n"
            f"   • ETH ({prices['ETH']:.6f} ETH)\n"
            f"   • USDT ({prices['USDT']:.2f} USDT)"
        )