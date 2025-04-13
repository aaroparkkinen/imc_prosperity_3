from datamodel import OrderDepth, UserId, TradingState, Order, Symbol
from typing import List
import string

class Trader:
    def __init__(self):
        # Limits for each product
        self.limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50
        }
        # History for momentum calculations for SQUID_INK and KELP
        self.squid_history: List[float] = []
        self.kelp_history: List[float] = []
        # Momentum thresholds (you can adjust these values based on testing and analysis)
        self.squid_momentum_threshold = 4.5
        self.kelp_momentum_threshold = 2

    """
    Rainforest resin oscillates within tight bands.
    Buy/Sell strategy for Rainforest Resin should be to buy when within 10% of the trailing floor, 
    and sell when within 10% of the trailing high.
    The floor can be established through visualisation, but better to determine it through data analysis.
    """
    def RainforestResinStrategy(self, position: int, order_depth: OrderDepth) -> List[Order]:
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

        # Check best ask from sell orders
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_volume = order_depth.sell_orders[best_ask]
            if best_ask <= floor_price + 1:
                # Determine how many can be bought without breaching the limit
                can_buy = limit - position
                qty_to_buy = min(can_buy, abs(best_ask_volume))
                if qty_to_buy > 0:
                    orders.append(Order(product, best_ask, qty_to_buy))

        # Check best bid from buy orders
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_volume = order_depth.buy_orders[best_bid]
            if best_bid >= ceiling_price - 1:
                # Determine how many can be sold without breaching the limit
                can_sell = limit + position  # if you are long, position > 0, so limit + position gives available volume to sell
                qty_to_sell = min(can_sell, best_bid_volume)
                if qty_to_sell > 0:
                    orders.append(Order(product, best_bid, -qty_to_sell))
        return orders

    """
    Momentum trading strategy for KELP based on the last 40 trailing time steps.
 
    Steps:
      1. Compute the mid-price from the order book (average of best bid and best ask).
      2. Append this mid-price to a trailing history list for KELP.
      3. If fewer than 40 data points exist, do nothing.
      4. Compute momentum = current mid-price - mid-price from 40 steps ago.
      5. If momentum > kelp threshold (upward momentum), buy at the ask price.
         If momentum < -kelp threshold (downward momentum), sell at the bid price.
      6. Respect the position limit of 50.
    """
    def KelpMomentumStrategy(self, position: int, order_depth: OrderDepth) -> List[Order]:

        product = "KELP"
        orders: List[Order] = []

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2

        # Update trailing history for KELP
        self.kelp_history.append(mid_price)
        if len(self.kelp_history) > 50:
            self.kelp_history = self.kelp_history[-50:]

        # Only trade once we have 40 data points.
        if len(self.kelp_history) < 50:
            return orders

        reference_price = self.kelp_history[0]
        momentum = mid_price - reference_price

        limit = 50

        # Upward momentum: signal to buy (trade at ask price)
        if momentum > self.kelp_momentum_threshold:
            can_buy = limit - position
            ask_volume = order_depth.sell_orders.get(best_ask, 0)
            qty = min(can_buy, abs(ask_volume))
            if qty > 0:
                orders.append(Order(product, best_ask, qty))

        # Downward momentum: signal to sell (trade at bid price)
        #elif momentum < -self.kelp_momentum_threshold:
        #    can_sell = limit + position
        #    bid_volume = order_depth.buy_orders.get(best_bid, 0)
        #    qty = min(can_sell, bid_volume)
        #    if qty > 0:
        #        orders.append(Order(product, best_bid, -qty))

        return orders

    """
    Momentum trading strategy for SQUID_INK based on the last 40 trailing time steps.
 
    Steps:
      1. Compute the mid-price from the order book (average of best bid and best ask).
      2. Append the mid-price to the trailing history list for SQUID_INK.
      3. If fewer than 40 data points exist, do nothing.
      4. Compute momentum = current mid-price - mid-price from 40 steps ago.
      5. If momentum > squid threshold (upward momentum), buy at ask price.
         If momentum < -squid threshold (downward momentum), sell at bid price.
      6. Respect the position limit of 50.
    """
    def SquidInkMomentumStrategy(self, position: int, order_depth: OrderDepth) -> List[Order]:

        product = "SQUID_INK"
        orders: List[Order] = []

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2

        # Update trailing history for SQUID_INK
        self.squid_history.append(mid_price)
        if len(self.squid_history) > 50:
            self.squid_history = self.squid_history[-50:]

        if len(self.squid_history) < 50:
            return orders

        reference_price = self.squid_history[0]
        momentum = mid_price - reference_price
        limit = 50

        # Upward momentum: signal to buy (trade at ask price)
        if momentum > self.squid_momentum_threshold:
            can_buy = limit - position
            ask_volume = order_depth.sell_orders.get(best_ask, 0)
            qty = min(can_buy, abs(ask_volume))
            if qty > 0:
                orders.append(Order(product, best_ask, qty))
        # Downward momentum: signal to sell (trade at bid price)

        #momentum < -self.squid_momentum_threshold:
        #    can_sell = limit + position
        #    bid_volume = order_depth.buy_orders.get(best_bid, 0)
        #    qty = min(can_sell, bid_volume)
        #    if qty > 0:
        #        orders.append(Order(product, best_bid, -qty))
        return orders


    def printTradingState(self, tradingState: TradingState):
        print(f"--- Trading State at timestamp {tradingState.timestamp} ---")
        print(f"Trader Data: {tradingState.traderData}")

        # Print current positions
        print("Positions:")
        for product, pos in tradingState.position.items():
            print(f"  {product}: {pos}")

        # Print order depths
        print("\nOrder Depths:")
        for symbol, depth in tradingState.order_depths.items():
            print(f"  Symbol: {symbol}")
            print(f"    Buy Orders:  {depth.buy_orders}")
            print(f"    Sell Orders: {depth.sell_orders}")

        # Print recent trades
        print("\nOwn Trades Since Last Update:")
        for symbol, trades in tradingState.own_trades.items():
            if trades:
                print(f"  {symbol}: {trades}")

        print("\nMarket Trades Since Last Update:")
        for symbol, trades in tradingState.market_trades.items():
            if trades:
                print(f"  {symbol}: {trades}")

        # Print observations
        print("\nObservations:")
        print(tradingState.observations)

        print("--- End of Trading State ---\n")
        return None


    def process_product(self, result, product, tradingState: TradingState):

        position = tradingState.position.get(product, 0)
        order_depth: OrderDepth = tradingState.order_depths[product]

        resulting_orders = []

        if product == "RAINFOREST_RESIN":
            print("Printing Trading State")
            self.printTradingState(tradingState)
            resulting_orders = self.RainforestResinStrategy(position, order_depth)
        elif product == "SQUID_INK":
            resulting_orders = self.SquidInkMomentumStrategy(position, order_depth)
        elif product == "KELP":
            resulting_orders = self.KelpMomentumStrategy(position, order_depth)

        return resulting_orders


    def run(self, state: TradingState) -> tuple[dict[Symbol, List[Order]], int, str]:
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        result = {}

        for product in state.order_depths:
            result[product] = self.process_product(result, product, state)


        traderData = "Hamish-Test"
        conversions = 1

        return result, conversions, traderData