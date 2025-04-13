import unittest
from datamodel import Trade

class TestTrade(unittest.TestCase):
    def test_trade_rainforest_resin(self):
        # Arrange
        symbol = "RAINFOREST_RESIN"
        price = 100
        quantity = 10
        buyer = "BUYER_ID"
        seller = "SELLER_ID"
        timestamp = 123456789

        # Act
        trade = Trade(symbol, price, quantity, buyer, seller, timestamp)

        # Assert
        self.assertEqual(trade.symbol, symbol)
        self.assertEqual(trade.price, price)
        self.assertEqual(trade.quantity, quantity)
        self.assertEqual(trade.buyer, buyer)
        self.assertEqual(trade.seller, seller)
        self.assertEqual(trade.timestamp, timestamp)

        # Verify the string representation
        expected_str = f"({symbol}, {buyer} << {seller}, {price}, {quantity}, {timestamp})"
        self.assertEqual(str(trade), expected_str)

if __name__ == '__main__':
    unittest.main()
