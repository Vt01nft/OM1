"""Store Assistant Provider for OM1.

Bounty #367: OM1 + Smart Assistant + Wallet Payments

This provider manages the product catalog and order processing,
integrating with the wallet payment provider for transactions.

Features:
- Product catalog management
- Order workflow handling
- Price calculation in multiple currencies
- Order history
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

try:
    from providers.singleton import singleton
except ImportError:

    def singleton(cls):
        """Singleton decorator fallback."""
        return cls


logger = logging.getLogger(__name__)


@singleton
class StoreAssistantProvider:
    """Store Assistant Provider for product ordering.

    Manages product catalog and order flow, integrating with
    wallet payments for transaction processing.

    Example:
        store = StoreAssistantProvider()
        products = store.get_products()
        order = store.create_order('coffee', 'latte')
        store.process_payment(order, 'SOL', wallet_provider)
    """

    # Product catalog
    PRODUCTS = {
        "coffee": {
            "name": "Coffee",
            "icon": "☕",
            "varieties": {
                "espresso": ("Espresso", 4.00),
                "americano": ("Americano", 4.50),
                "latte": ("Latte", 5.50),
                "cappuccino": ("Cappuccino", 5.50),
                "mocha": ("Mocha", 6.00),
                "cold brew": ("Cold Brew", 5.00),
            },
        },
        "tea": {
            "name": "Tea",
            "icon": "🍵",
            "varieties": {
                "green": ("Green Tea", 3.50),
                "green tea": ("Green Tea", 3.50),
                "black": ("Black Tea", 3.50),
                "black tea": ("Black Tea", 3.50),
                "chai": ("Chai Latte", 4.50),
                "chai latte": ("Chai Latte", 4.50),
                "matcha": ("Matcha", 5.00),
            },
        },
        "pizza": {
            "name": "Pizza",
            "icon": "🍕",
            "varieties": {
                "margherita": ("Margherita", 14.00),
                "margarita": ("Margherita", 14.00),
                "pepperoni": ("Pepperoni", 16.00),
                "veggie": ("Veggie", 15.00),
                "vegetable": ("Veggie", 15.00),
                "hawaiian": ("Hawaiian", 16.00),
            },
        },
        "burger": {
            "name": "Burger",
            "icon": "🍔",
            "varieties": {
                "classic": ("Classic Burger", 10.00),
                "cheese": ("Cheeseburger", 11.00),
                "cheeseburger": ("Cheeseburger", 11.00),
                "bacon": ("Bacon Burger", 13.00),
                "veggie": ("Veggie Burger", 11.00),
                "veggie burger": ("Veggie Burger", 11.00),
            },
        },
    }

    def __init__(self):
        """Initialize the Store Assistant Provider."""
        self.current_order = None
        self.order_history = []
        logger.info("StoreAssistantProvider initialized")

    # ─────────────────────────────────────────────────────────────
    # Product Catalog
    # ─────────────────────────────────────────────────────────────

    def get_products(self) -> List[str]:
        """Get list of available product categories."""
        return list(self.PRODUCTS.keys())

    def get_product_info(self, product: str) -> Optional[Dict[str, Any]]:
        """Get information about a product.

        Args:
            product: Product name (e.g., 'coffee').

        Returns
        -------
            Product info dict or None.
        """
        return self.PRODUCTS.get(product.lower())

    def get_varieties(self, product: str) -> List[Tuple[str, float]]:
        """Get varieties for a product.

        Args:
            product: Product name.

        Returns
        -------
            List of (variety_name, price) tuples.
        """
        info = self.get_product_info(product)
        if not info:
            return []

        # Get unique varieties
        seen = set()
        varieties = []
        for key, (name, price) in info["varieties"].items():
            if name not in seen:
                varieties.append((name, price))
                seen.add(name)

        return varieties

    def get_variety_names(self, product: str) -> List[str]:
        """Get just the variety names for a product."""
        return [name for name, _ in self.get_varieties(product)]

    def find_variety(
        self, product: str, variety_text: str
    ) -> Optional[Tuple[str, float]]:
        """Find a variety by text match.

        Args:
            product: Product name.
            variety_text: User input text.

        Returns
        -------
            (variety_name, price) or None.
        """
        info = self.get_product_info(product)
        if not info:
            return None

        text = variety_text.lower()

        # Try exact match first
        if text in info["varieties"]:
            name, price = info["varieties"][text]
            return (name, price)

        # Try partial match
        for key, (name, price) in info["varieties"].items():
            if key in text or text in key:
                return (name, price)

        return None

    def detect_product(self, text: str) -> Optional[str]:
        """Detect product from text.

        Args:
            text: User input text.

        Returns
        -------
            Product name or None.
        """
        text_lower = text.lower()
        for product in self.PRODUCTS:
            if product in text_lower:
                return product
        return None

    # ─────────────────────────────────────────────────────────────
    # Order Management
    # ─────────────────────────────────────────────────────────────

    def create_order(self, product: str, variety: str) -> Optional[Dict[str, Any]]:
        """Create a new order.

        Args:
            product: Product category.
            variety: Variety name.

        Returns
        -------
            Order dict or None.
        """
        variety_info = self.find_variety(product, variety)
        if not variety_info:
            logger.warning(f"Variety not found: {product}/{variety}")
            return None

        name, price = variety_info
        product_info = self.get_product_info(product)

        self.current_order = {
            "product": product,
            "variety": name,
            "price_usd": price,
            "icon": product_info.get("icon", "🛒"),
            "status": "pending",
        }

        logger.info(f"Order created: {name} (${price:.2f})")
        return self.current_order

    def calculate_prices(
        self, usd_amount: float, sol_price: float, eth_price: float
    ) -> Dict[str, float]:
        """Calculate payment amounts in different tokens.

        Args:
            usd_amount: Amount in USD.
            sol_price: Current SOL price in USD.
            eth_price: Current ETH price in USD.

        Returns
        -------
            Dictionary of token -> amount.
        """
        return {
            "SOL": usd_amount / sol_price,
            "ETH": usd_amount / eth_price,
            "USDC": usd_amount,
            "USDT": usd_amount,
        }

    def process_payment(
        self,
        order: Dict[str, Any],
        token: str,
        amount: float,
        wallet_provider: Any,
    ) -> Dict[str, Any]:
        """Process payment for an order.

        Args:
            order: Order dict.
            token: Payment token.
            amount: Payment amount.
            wallet_provider: WalletPaymentProvider instance.

        Returns
        -------
            Transaction result.
        """
        if not order:
            return {"success": False, "error": "No order"}

        result = wallet_provider.pay_store(
            token=token, amount=amount, item=order["variety"]
        )

        if result.get("success"):
            order["status"] = "completed"
            order["payment_token"] = token
            order["payment_amount"] = amount
            self.order_history.append(order)
            self.current_order = None
            logger.info(f"Order completed: {order['variety']}")
        else:
            order["status"] = "failed"
            order["error"] = result.get("error")
            logger.error(f"Order failed: {result.get('error')}")

        return result

    def cancel_order(self) -> None:
        """Cancel the current order."""
        if self.current_order:
            self.current_order["status"] = "cancelled"
            logger.info("Order cancelled")
            self.current_order = None

    def get_current_order(self) -> Optional[Dict[str, Any]]:
        """Get the current pending order."""
        return self.current_order

    def get_order_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent order history."""
        return self.order_history[-limit:] if self.order_history else []

    # ─────────────────────────────────────────────────────────────
    # Formatted Output
    # ─────────────────────────────────────────────────────────────

    def format_menu(self, product: str) -> str:
        """Format menu for display.

        Args:
            product: Product name.

        Returns
        -------
            Formatted menu string.
        """
        info = self.get_product_info(product)
        if not info:
            return f"Product not found: {product}"

        lines = [f"\n   {info['icon']} {info['name']} Menu:"]
        lines.append("   " + "─" * 35)

        for name, price in self.get_varieties(product):
            lines.append(f"      • {name:15} ${price:.2f}")

        return "\n".join(lines)

    def format_order_summary(
        self, order: Dict[str, Any], prices: Dict[str, float]
    ) -> str:
        """Format order summary for display.

        Args:
            order: Order dict.
            prices: Token prices dict.

        Returns
        -------
            Formatted summary string.
        """
        return f"""
   ┌─────────────────────────────────────┐
   │  🧾 ORDER SUMMARY                  │
   ├─────────────────────────────────────┤
   │  Item: {order['variety']:20}      │
   │  Price: ${order['price_usd']:.2f}                        │
   ├─────────────────────────────────────┤
   │  💳 Payment Options:               │
   │  • SOL:  {prices['SOL']:.6f} SOL              │
   │  • ETH:  {prices['ETH']:.6f} ETH              │
   │  • USDC: {prices['USDC']:.2f} USDC                │
   │  • USDT: {prices['USDT']:.2f} USDT                │
   └─────────────────────────────────────┘
"""

    # ─────────────────────────────────────────────────────────────
    # Status
    # ─────────────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Get provider status."""
        return {
            "products_available": len(self.PRODUCTS),
            "current_order": self.current_order,
            "total_orders": len(self.order_history),
        }

    def __repr__(self) -> str:
        """Return string representation."""
        return f"StoreAssistantProvider(products={len(self.PRODUCTS)})"
