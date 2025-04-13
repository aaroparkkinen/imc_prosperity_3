from unittest import TestCase
from datamodel import Listing, OrderDepth, Trade, TradingState, Observation
from round1LoggingTest import Trader

from unittest import TestCase
from datamodel import Listing, OrderDepth, Trade, TradingState, Observation
from round1LoggingTest import Trader

class TestTrader(TestCase):
    def synthesize_trading_state(self):
        order_depths = {}

        # CROISSANTS
        croissants_depth = OrderDepth()
        croissants_depth.buy_orders = {4274: 127}
        croissants_depth.sell_orders = {4275: -39, 4276: -88}
        order_depths["CROISSANTS"] = croissants_depth

        # DJEMBES
        djembes_depth = OrderDepth()
        djembes_depth.buy_orders = {13410: 20, 13409: 37}
        djembes_depth.sell_orders = {13411: -57}
        order_depths["DJEMBES"] = djembes_depth

        # JAMS
        jams_depth = OrderDepth()
        jams_depth.buy_orders = {6541: 88, 6540: 160}
        jams_depth.sell_orders = {6542: -88, 6543: -160}
        order_depths["JAMS"] = jams_depth

        # KELP
        kelp_depth = OrderDepth()
        kelp_depth.buy_orders = {2031: 22}
        kelp_depth.sell_orders = {2035: -22}
        order_depths["KELP"] = kelp_depth

        # PICNIC_BASKET1
        pb1_depth = OrderDepth()
        pb1_depth.buy_orders = {58697: 14, 58696: 20}
        pb1_depth.sell_orders = {58708: -34}
        order_depths["PICNIC_BASKET1"] = pb1_depth

        # PICNIC_BASKET2
        pb2_depth = OrderDepth()
        pb2_depth.buy_orders = {30256: 34}
        pb2_depth.sell_orders = {30262: -14, 30263: -20}
        order_depths["PICNIC_BASKET2"] = pb2_depth

        # RAINFOREST_RESIN (will be modified in our test scenarios)
        resin_depth = OrderDepth()
        resin_depth.buy_orders = {9992: 25}
        resin_depth.sell_orders = {10008: -25}
        order_depths["RAINFOREST_RESIN"] = resin_depth

        # SQUID_INK
        ink_depth = OrderDepth()
        ink_depth.buy_orders = {1838: 22}
        ink_depth.sell_orders = {1842: -22}
        order_depths["SQUID_INK"] = ink_depth

        # Market trades from the log snippet
        market_trades = {
            "RAINFOREST_RESIN": [Trade("RAINFOREST_RESIN", 9999, 1, buyer=None, seller=None, timestamp=100)],
            "SQUID_INK":        [Trade("SQUID_INK", 1841, 1, buyer=None, seller=None, timestamp=300)],
            "KELP":             [Trade("KELP", 2032, 1, buyer=None, seller=None, timestamp=300)],
            "CROISSANTS":       [Trade("CROISSANTS", 4276, 8, buyer=None, seller=None, timestamp=200)],
            "JAMS":             [Trade("JAMS", 6543, 7, buyer=None, seller=None, timestamp=200)],
            "DJEMBES": [],
            "PICNIC_BASKET1": [],
            "PICNIC_BASKET2": []
        }

        # No own trades shown in the log
        own_trades = {
            "CROISSANTS": [],
            "DJEMBES": [],
            "JAMS": [],
            "KELP": [],
            "PICNIC_BASKET1": [],
            "PICNIC_BASKET2": [],
            "RAINFOREST_RESIN": [],
            "SQUID_INK": []
        }

        # Empty positions and observations
        position = {}
        observations = Observation(plainValueObservations={}, conversionObservations={})

        # Construct and return the TradingState
        trading_state = TradingState(
            traderData="Hamish-Test",
            timestamp=500,
            listings={},  # No listings shown in log; add if needed
            order_depths=order_depths,
            own_trades=own_trades,
            market_trades=market_trades,
            position=position,
            observations=observations
        )

        return trading_state

    def test_rainforest_resin_strategy(self):
        trader = Trader()
        trading_state = self.synthesize_trading_state()

        """
        SCENARIO 1: 
          Modify RAINFOREST_RESIN's order_depth so best ask <= floor_price + 1 (9998).
          Expect a buy order.
        """
        position_buy = 0
        trading_state.order_depths["RAINFOREST_RESIN"].sell_orders = {9998: -10}  # best ask at 9998
        orders_buy = trader.RainforestResinStrategy(
            position_buy,
            trading_state.order_depths["RAINFOREST_RESIN"]
        )

        self.assertEqual(len(orders_buy), 1, "Should produce 1 buy order")
        self.assertEqual(orders_buy[0].symbol, "RAINFOREST_RESIN")
        self.assertEqual(orders_buy[0].price, 9998)
        self.assertEqual(orders_buy[0].quantity, 10, "Should buy all 10 units")

        """
        SCENARIO 2:
          Modify RAINFOREST_RESIN's order_depth so best bid >= ceiling_price - 1 (10002).
          Expect a sell order.
        """
        position_sell = 10
        trading_state.order_depths["RAINFOREST_RESIN"].buy_orders = {10002: 5}  # best bid at 10002
        # Also reset sell_orders so it doesn't trigger scenario 1 logic
        trading_state.order_depths["RAINFOREST_RESIN"].sell_orders = {}

        orders_sell = trader.RainforestResinStrategy(
            position_sell,
            trading_state.order_depths["RAINFOREST_RESIN"]
        )

        self.assertEqual(len(orders_sell), 1, "Should produce 1 sell order")
        self.assertEqual(orders_sell[0].symbol, "RAINFOREST_RESIN")
        self.assertEqual(orders_sell[0].price, 10002)
        self.assertEqual(orders_sell[0].quantity, -5, "Should sell 5 units")
