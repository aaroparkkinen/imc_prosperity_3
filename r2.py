from typing import List, Any

from jsonpickle import json

from datamodel import OrderDepth, TradingState, Order, Symbol, Listing, \
    Trade, Observation, ProsperityEncoder


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]],
        conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData,
                                                             max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[
        Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[
        list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append(
                [listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> \
    dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders,
                                  order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[
        list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[
        list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


class Trader:
    def __init__(self):
        # Position limits for all products
        self.limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
            "CROISSANTS": 50,
            "JAMS": 50,
            "DJEMBES": 50,
            "PICNIC_BASKET1": 50,
            "PICNIC_BASKET2": 50
        }
        # Histories for momentum-based strategies
        self.squid_history: List[float] = []
        self.kelp_history: List[float] = []

        # Momentum thresholds
        self.squid_momentum_threshold = 4.5
        self.kelp_momentum_threshold = 2

    ############################################################################
    # Existing / Example Strategies
    ############################################################################

    def RainforestResinStrategy(self, position: int, order_depth: OrderDepth) -> \
    List[Order]:
        """
        Buys if best ask <= 9997 + 1,
        sells if best bid >= 10003 - 1,
        respecting a position limit of 50.
        """
        product = "RAINFOREST_RESIN"
        floor_price = 9997
        ceiling_price = 10003
        limit = self.limits[product]
        orders: List[Order] = []

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_volume = order_depth.sell_orders[best_ask]
            if best_ask <= floor_price + 1:
                can_buy = limit - position
                qty_to_buy = min(can_buy, abs(best_ask_volume))
                if qty_to_buy > 0:
                    orders.append(Order(product, best_ask, qty_to_buy))

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_volume = order_depth.buy_orders[best_bid]
            if best_bid >= ceiling_price - 1:
                can_sell = limit + position
                qty_to_sell = min(can_sell, best_bid_volume)
                if qty_to_sell > 0:
                    orders.append(Order(product, best_bid, -qty_to_sell))

        return orders

    def KelpMomentumStrategy(self, position: int, order_depth: OrderDepth) -> \
    List[Order]:
        product = "KELP"
        limit = self.limits[product]
        orders: List[Order] = []

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2

        self.kelp_history.append(mid_price)
        if len(self.kelp_history) > 50:
            self.kelp_history = self.kelp_history[-50:]

        # Only trade once we have 40 data points
        if len(self.kelp_history) < 50:
            return orders

        reference_price = self.kelp_history[0]
        momentum = mid_price - reference_price

        # Upward momentum => buy
        if momentum > self.kelp_momentum_threshold:
            can_buy = limit - position
            ask_volume = order_depth.sell_orders.get(best_ask, 0)
            qty = min(can_buy, abs(ask_volume))
            if qty > 0:
                orders.append(Order(product, best_ask, qty))

        return orders

    def SquidInkMomentumStrategy(self, position: int,
        order_depth: OrderDepth) -> List[Order]:
        product = "SQUID_INK"
        limit = self.limits[product]
        orders: List[Order] = []

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2

        self.squid_history.append(mid_price)
        if len(self.squid_history) > 50:
            self.squid_history = self.squid_history[-50:]

        if len(self.squid_history) < 50:
            return orders

        reference_price = self.squid_history[0]
        momentum = mid_price - reference_price

        # Upward momentum => buy
        if momentum > self.squid_momentum_threshold:
            can_buy = limit - position
            ask_volume = order_depth.sell_orders.get(best_ask, 0)
            qty = min(can_buy, abs(ask_volume))
            if qty > 0:
                orders.append(Order(product, best_ask, qty))

        return orders

    ############################################################################
    # Simple Range Strategies for Croissants, Jams, Djembes (if you want them)
    ############################################################################

    def CroissantsStrategy(self, position: int, order_depth: OrderDepth) -> \
    List[Order]:
        product = "CROISSANTS"
        floor_price = 4270
        ceiling_price = 4280
        limit = self.limits[product]
        orders: List[Order] = []

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_vol = order_depth.sell_orders[best_ask]
            if best_ask <= floor_price + 1:
                can_buy = limit - position
                qty_to_buy = min(can_buy, abs(best_ask_vol))
                if qty_to_buy > 0:
                    orders.append(Order(product, best_ask, qty_to_buy))

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_vol = order_depth.buy_orders[best_bid]
            if best_bid >= ceiling_price - 1:
                can_sell = limit + position
                qty_to_sell = min(can_sell, best_bid_vol)
                if qty_to_sell > 0:
                    orders.append(Order(product, best_bid, -qty_to_sell))

        return orders

    def JamsStrategy(self, position: int, order_depth: OrderDepth) -> List[
        Order]:
        product = "JAMS"
        floor_price = 6540
        ceiling_price = 6550
        limit = self.limits[product]
        orders: List[Order] = []

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_vol = order_depth.sell_orders[best_ask]
            if best_ask <= floor_price + 1:
                can_buy = limit - position
                qty_to_buy = min(can_buy, abs(best_ask_vol))
                if qty_to_buy > 0:
                    orders.append(Order(product, best_ask, qty_to_buy))

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_vol = order_depth.buy_orders[best_bid]
            if best_bid >= ceiling_price - 1:
                can_sell = limit + position
                qty_to_sell = min(can_sell, best_bid_vol)
                if qty_to_sell > 0:
                    orders.append(Order(product, best_bid, -qty_to_sell))

        return orders

    def DjembesStrategy(self, position: int, order_depth: OrderDepth) -> List[
        Order]:
        product = "DJEMBES"
        floor_price = 13400
        ceiling_price = 13420
        limit = self.limits[product]
        orders: List[Order] = []

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_vol = order_depth.sell_orders[best_ask]
            if best_ask <= floor_price + 1:
                can_buy = limit - position
                qty_to_buy = min(can_buy, abs(best_ask_vol))
                if qty_to_buy > 0:
                    orders.append(Order(product, best_ask, qty_to_buy))

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_vol = order_depth.buy_orders[best_bid]
            if best_bid >= ceiling_price - 1:
                can_sell = limit + position
                qty_to_sell = min(can_sell, best_bid_vol)
                if qty_to_sell > 0:
                    orders.append(Order(product, best_bid, -qty_to_sell))

        return orders

    ############################################################################
    # Picnic Basket Arbitrage
    ############################################################################

    def PicnicBasket1ArbStrategy(
        self,
        position_basket: int,
        tradingState: TradingState
    ) -> List[Order]:
        """
        Picnic Basket 1 = 6 CROISSANTS + 3 JAMS + 1 DJEMBES
        Arbitrage conditions:
          1) If best_ask_basket1 < 6*best_bid_croissants + 3*best_bid_jams + 1*best_bid_djembes,
             buy basket, sell constituents
          2) If best_bid_basket1 > 6*best_ask_croissants + 3*best_ask_jams + 1*best_ask_djembes,
             buy constituents, sell basket
        Ignores volumes for brevity; you may incorporate them to limit order sizes.
        """
        product = "PICNIC_BASKET1"
        orders: List[Order] = []
        limit_basket = self.limits[product]

        # Gather each relevant order depth
        depth_basket = tradingState.order_depths.get("PICNIC_BASKET1")
        depth_croissants = tradingState.order_depths.get("CROISSANTS")
        depth_jams = tradingState.order_depths.get("JAMS")
        depth_djembes = tradingState.order_depths.get("DJEMBES")

        # Ensure we have all the required order books
        if not (
            depth_basket and depth_croissants and depth_jams and depth_djembes):
            return orders

        # Helper: best ask & best bid for each item
        def best_ask(depth: OrderDepth):
            return min(depth.sell_orders.keys()) if depth.sell_orders else None

        def best_bid(depth: OrderDepth):
            return max(depth.buy_orders.keys()) if depth.buy_orders else None

        ask_basket = best_ask(depth_basket)
        bid_basket = best_bid(depth_basket)
        ask_croissants = best_ask(depth_croissants)
        bid_croissants = best_bid(depth_croissants)
        ask_jams = best_ask(depth_jams)
        bid_jams = best_bid(depth_jams)
        ask_djembes = best_ask(depth_djembes)
        bid_djembes = best_bid(depth_djembes)

        # Check we have valid prices
        if not (
            ask_basket and bid_basket and ask_croissants and bid_croissants and
            ask_jams and bid_jams and ask_djembes and bid_djembes):
            return orders

        # Calculate the sum of the constituents (for 1 unit of basket)
        # Sell constituents = use best BID for each
        basket_contents_bid_value = 6 * bid_croissants + 3 * bid_jams + 1 * bid_djembes
        # Buy constituents = use best ASK for each
        basket_contents_ask_value = 6 * ask_croissants + 3 * ask_jams + 1 * ask_djembes

        # 1) Buy the basket at ask, sell the constituents at their bids
        #    if ask_basket < sum of constituent bids
        if ask_basket < basket_contents_bid_value:
            # We'll do 1 basket for simplicity (or you could do multiple)
            can_buy_basket = limit_basket - position_basket
            if can_buy_basket > 0:
                # Place an order to buy +1 basket
                ask_volume_basket = depth_basket.sell_orders[ask_basket]
                qty_basket = min(can_buy_basket, abs(ask_volume_basket))
                if qty_basket > 0:
                    orders.append(Order(product, ask_basket, qty_basket))
                # You would also place SELL orders on the constituents
                # (CROISSANTS, JAMS, DJEMBES) if you want to realize the arb
                # Example (selling 6 CROISSANTS * qty_basket):
                # This requires you to track positions in CROISSANTS, etc.
                # For brevity we omit those orders. You can incorporate them
                # if your environment supports multi-product trades.

        # 2) Buy the constituents at ask, sell the basket at bid
        #    if bid_basket > sum of constituent asks
        if bid_basket > basket_contents_ask_value:
            # We'll do 1 basket for simplicity
            can_sell_basket = self.limits["PICNIC_BASKET1"] + position_basket
            if can_sell_basket > 0:
                bid_volume_basket = depth_basket.buy_orders[bid_basket]
                qty_basket = min(can_sell_basket, bid_volume_basket)
                if qty_basket > 0:
                    # Place an order to SELL the basket
                    orders.append(Order(product, bid_basket, -qty_basket))
                # Similarly, you'd place BUY orders for the constituents
                # (6 CROISSANTS, 3 JAMS, 1 DJEMBES). Omitted for brevity.

        return orders

    def PicnicBasket2ArbStrategy(
        self,
        position_basket: int,
        tradingState: TradingState
    ) -> List[Order]:
        """
        Picnic Basket 2 = 4 CROISSANTS + 2 JAMS
        Similar arbitrage concept:
          1) If best_ask_basket2 < 4*best_bid_croissants + 2*best_bid_jams => buy basket, sell croissants+jams
          2) If best_bid_basket2 > 4*best_ask_croissants + 2*best_ask_jams => buy croissants+jams, sell basket
        """
        product = "PICNIC_BASKET2"
        orders: List[Order] = []
        limit_basket = self.limits[product]

        depth_basket = tradingState.order_depths.get("PICNIC_BASKET2")
        depth_croissants = tradingState.order_depths.get("CROISSANTS")
        depth_jams = tradingState.order_depths.get("JAMS")

        if not (depth_basket and depth_croissants and depth_jams):
            return orders

        def best_ask(depth: OrderDepth):
            return min(depth.sell_orders.keys()) if depth.sell_orders else None

        def best_bid(depth: OrderDepth):
            return max(depth.buy_orders.keys()) if depth.buy_orders else None

        ask_basket = best_ask(depth_basket)
        bid_basket = best_bid(depth_basket)
        ask_croissants = best_ask(depth_croissants)
        bid_croissants = best_bid(depth_croissants)
        ask_jams = best_ask(depth_jams)
        bid_jams = best_bid(depth_jams)

        if not (
            ask_basket and bid_basket and ask_croissants and bid_croissants and ask_jams and bid_jams):
            return orders

        basket_contents_bid_value = 4 * bid_croissants + 2 * bid_jams
        basket_contents_ask_value = 4 * ask_croissants + 2 * ask_jams

        # 1) Buy basket at ask, sell constituents
        if ask_basket < basket_contents_bid_value:
            can_buy_basket = limit_basket - position_basket
            if can_buy_basket > 0:
                ask_volume_basket = depth_basket.sell_orders[ask_basket]
                qty_basket = min(can_buy_basket, abs(ask_volume_basket))
                if qty_basket > 0:
                    orders.append(Order(product, ask_basket, qty_basket))
                # Omit SELL orders for croissants/jams for brevity

        # 2) Buy constituents, sell basket at bid
        if bid_basket > basket_contents_ask_value:
            can_sell_basket = limit_basket + position_basket
            if can_sell_basket > 0:
                bid_volume_basket = depth_basket.buy_orders[bid_basket]
                qty_basket = min(can_sell_basket, bid_volume_basket)
                if qty_basket > 0:
                    orders.append(Order(product, bid_basket, -qty_basket))
                # Omit BUY orders for croissants/jams for brevity

        return orders

    ############################################################################
    # Logging
    ############################################################################

    def printTradingState(self, tradingState: TradingState):
        print(f"--- Trading State at timestamp {tradingState.timestamp} ---")
        print(f"Trader Data: {tradingState.traderData}")

        print("Positions:")
        for product, pos in tradingState.position.items():
            print(f"  {product}: {pos}")

        print("\nOrder Depths:")
        for symbol, depth in tradingState.order_depths.items():
            print(f"  Symbol: {symbol}")
            print(f"    Buy Orders:  {depth.buy_orders}")
            print(f"    Sell Orders: {depth.sell_orders}")

        print("\nOwn Trades Since Last Update:")
        for symbol, trades in tradingState.own_trades.items():
            if trades:
                print(f"  {symbol}: {trades}")

        print("\nMarket Trades Since Last Update:")
        for symbol, trades in tradingState.market_trades.items():
            if trades:
                print(f"  {symbol}: {trades}")

        print("\nObservations:")
        print(tradingState.observations)
        print("--- End of Trading State ---\n")
        return None

    ############################################################################
    # Main run / Orchestration
    ############################################################################

    def process_product(self, result, product, tradingState: TradingState) -> \
    List[Order]:
        """
        Decide how to trade each product.
        We'll call the existing or new strategies as appropriate.
        For Picnic Baskets, use the arb strategies.
        """
        position = tradingState.position.get(product, 0)
        order_depth: OrderDepth = tradingState.order_depths[product]
        orders: List[Order] = []

        if product == "RAINFOREST_RESIN":
            # Example: print state when we handle RAINFOREST_RESIN
            self.printTradingState(tradingState)
            orders = self.RainforestResinStrategy(position, order_depth)

        elif product == "SQUID_INK":
            orders = self.SquidInkMomentumStrategy(position, order_depth)

        elif product == "KELP":
            orders = self.KelpMomentumStrategy(position, order_depth)

        elif product == "CROISSANTS":
            orders = self.CroissantsStrategy(position, order_depth)

        elif product == "JAMS":
            orders = self.JamsStrategy(position, order_depth)

        elif product == "DJEMBES":
            orders = self.DjembesStrategy(position, order_depth)

        elif product == "PICNIC_BASKET1":
            # Arbitrage strategy for Basket1
            orders = self.PicnicBasket1ArbStrategy(position, tradingState)

        elif product == "PICNIC_BASKET2":
            # Arbitrage strategy for Basket2
            orders = self.PicnicBasket2ArbStrategy(position, tradingState)

        return orders

    def run(self, state: TradingState) -> tuple[
        dict[Symbol, List[Order]], int, str]:
        """
        1) For each product in state.order_depths, decide how to trade.
        2) Return (dict of symbol->List[Order], conversions used, updated traderData).
        """
        print(f"traderData: {state.traderData}")
        print(f"Observations: {state.observations}")

        result = {}
        for product in state.order_depths:
            result[product] = self.process_product(result, product, state)

        conversions = 0
        traderData = "Hamish-Test"

        logger.flush(state, result, conversions, traderData)

        return (result, conversions, traderData)
