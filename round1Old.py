from datamodel import OrderDepth, UserId, TradingState, Order, Symbol
from typing import List
import string

class Trader:
    def __init__(self):
        limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50
        }

    """
    Rainforest resin oscillates within tight bands.  Buy/Sell strategy for Rainforest Resin should be to buy when within 10% of the trailing floor, and sell when within 10% of the trailing high.
    The floor can be established through visualisation, but better to determine it through data analysis.
    """
    def RainforestResinStrategy(self, position, order_depth) -> List[Order]:
        """
        Buys if best ask <= floor + 1,
        sells if best bid >= ceiling - 1,
        respecting a position limit of 50.
        """
        product = "RAINFOREST_RESIN"
        floor_price = 9997
        ceiling_price = 10003
        limit = 50


        orders: List[Order] = []

        # Check best ask
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_volume = order_depth.sell_orders[best_ask]

            if best_ask <= floor_price + 1:
                # Buy up to the limit
                can_buy = limit - position
                qty_to_buy = min(can_buy,
                                 abs(best_ask_volume))  # abs() if volume is negative
                if qty_to_buy > 0:
                    orders.append(Order(product, best_ask, qty_to_buy))

        # Check best bid
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_volume = order_depth.buy_orders[best_bid]
            if best_bid >= ceiling_price - 1:
                # Sell up to the limit
                can_sell = limit + position  # if you have 10, can_sell=60 => 10 + 50
                qty_to_sell = min(can_sell, best_bid_volume)
                if qty_to_sell > 0:
                    # Negative quantity for a sell
                    orders.append(Order(product, best_bid, -qty_to_sell))

        return orders

    def KelpStrategy(self):
        return None

    def SquidInkStrategy(self):
        return None

    def run(self, state: TradingState) -> tuple[
        dict[dict[Symbol, OrderDepth], List[Order]], int, str]:
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        result = {}

        for product in state.order_depths:

            position = state.position.get(product, 0)
            order_depth = state.order_depths[product]

            if product == "RAINFOREST_RESIN":
                result[product] = self.RainforestResinStrategy(position, order_depth)


        print(result)


        traderData = "Hamish-Test"

        # Sample conversion request. Check more details below.
        conversions = 1

        return result, conversions, traderData
