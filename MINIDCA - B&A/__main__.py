from binance_common.configuration import ConfigurationRestAPI
from binance_sdk_spot.spot import Spot
from dotenv import load_dotenv
import sqlite3
import logging
import asyncio
import os
import requests
import signal
import sys
from decimal import Decimal, ROUND_DOWN
from datetime import datetime, date
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)

# Get API credentials from environment variables
api_key = os.getenv("API_KEY")
api_secret = os.getenv("API_SECRET")

if not api_key or not api_secret:
    raise ValueError("API_KEY and API_SECRET must be set in .env file")

# Configuration for REST API
configuration_rest_api = ConfigurationRestAPI(api_key=api_key, api_secret=api_secret)
client = Spot(config_rest_api=configuration_rest_api)

# Database functions
def init_database():
    """Initialize SQLite database for DCA logging"""
    conn = sqlite3.connect('dca_bot.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS dca_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            quote_spent REAL NOT NULL,
            executed_qty REAL NOT NULL,
            average_price REAL NOT NULL,
            commission REAL,
            commission_asset TEXT,
            order_id TEXT,
            dry_run BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def log_dca_order(execution_analysis, dry_run=False):
    """Log DCA order to SQLite database"""
    try:
        conn = sqlite3.connect('dca_bot.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO dca_orders 
            (date, symbol, quote_spent, executed_qty, average_price, 
             commission, commission_asset, order_id, dry_run)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            date.today().isoformat(),
            execution_analysis['symbol'],
            execution_analysis['cumulative_quote_qty'],
            execution_analysis['executed_qty'],
            execution_analysis['average_price'],
            execution_analysis['total_commission'],
            execution_analysis['commission_asset'],
            str(execution_analysis['order_id']),
            dry_run
        ))
        
        conn.commit()
        conn.close()
        print(f"✓ Order logged to database")
        return True
        
    except Exception as e:
        print(f"Error logging to database: {e}")
        return False

def get_daily_stats(target_date=None):
    """Get daily spending and order count"""
    if target_date is None:
        target_date = date.today()
    
    try:
        conn = sqlite3.connect('dca_bot.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as order_count,
                COALESCE(SUM(quote_spent), 0) as total_spent
            FROM dca_orders 
            WHERE date = ? AND dry_run = FALSE
        ''', (target_date.isoformat(),))
        
        result = cursor.fetchone()
        conn.close()
        
        return {
            'order_count': result[0],
            'total_spent': result[1],
            'date': target_date
        }
        
    except Exception as e:
        print(f"Error getting daily stats: {e}")
        return {'order_count': 0, 'total_spent': 0.0, 'date': target_date}

async def send_discord_alert(execution_analysis, dry_run=False):
    """Send Discord webhook notification"""
    discord_webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    
    if not discord_webhook_url:
        print("Discord webhook URL not configured")
        return False
    
    try:
        status_emoji = "🧪" if dry_run else "✅"
        mode_text = "DRY RUN" if dry_run else "LIVE"
        
        # Discord embed format
        embed = {
            "title": f"{status_emoji} DCA Bot - {mode_text}",
            "color": 0x00ff00 if not dry_run else 0xffaa00,  # Green for live, orange for dry run
            "fields": [
                {
                    "name": "📊 Symbol",
                    "value": execution_analysis['symbol'],
                    "inline": True
                },
                {
                    "name": "💰 Quantity",
                    "value": f"{execution_analysis['executed_qty']:.8f} {execution_analysis['symbol'].replace('USDT', '')}",
                    "inline": True
                },
                {
                    "name": "💵 Cost",
                    "value": f"${execution_analysis['cumulative_quote_qty']:.2f} USDT",
                    "inline": True
                },
                {
                    "name": "📈 Price",
                    "value": f"${execution_analysis['average_price']:.2f}",
                    "inline": True
                },
                {
                    "name": "💸 Commission",
                    "value": f"{execution_analysis['total_commission']:.8f} {execution_analysis['commission_asset']}",
                    "inline": True
                },
                {
                    "name": "🆔 Order ID",
                    "value": str(execution_analysis['order_id']),
                    "inline": True
                }
            ],
            "footer": {
                "text": f"Executed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            }
        }
        
        payload = {
            "embeds": [embed]
        }
        
        response = requests.post(discord_webhook_url, json=payload, timeout=10)
        
        if response.status_code == 204:  # Discord webhooks return 204 on success
            print("✓ Discord alert sent")
            return True
        else:
            print(f"Failed to send Discord alert: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"Error sending Discord alert: {e}")
        return False

async def get_current_price(symbol="BTCEUR"):
    """Get current price for a symbol"""
    try:
        response = client.rest_api.ticker_price(symbol=symbol)
        price_data = response.data()
        
        # Access the actual_instance which contains the real data
        actual_data = price_data.actual_instance
        
        return {
            "symbol": actual_data.symbol,
            "price": float(actual_data.price)
        }
        
    except Exception as e:
        print(f"Error getting current price: {e}")
        return None

async def get_exchange_info(symbol="BTCEUR"):
    """Get trading rules for a symbol"""
    try:
        response = client.rest_api.exchange_info(symbol=symbol)
        exchange_info = response.data()
        
        # No actual_instance here - data is directly accessible
        symbol_info = exchange_info.symbols[0]
        
        lot_size = None
        min_notional = None
        price_filter = None
        
        for filter_item in symbol_info.filters:
            filter_type = filter_item.filter_type
            
            if filter_type == "LOT_SIZE":
                lot_size = {
                    "minQty": float(filter_item.min_qty),
                    "maxQty": float(filter_item.max_qty),
                    "stepSize": float(filter_item.step_size)
                }
            elif filter_type == "NOTIONAL":  # Note: it's "NOTIONAL" not "MIN_NOTIONAL"
                min_notional = {
                    "minNotional": float(filter_item.min_notional)
                }
            elif filter_type == "PRICE_FILTER":
                price_filter = {
                    "minPrice": float(filter_item.min_price),
                    "maxPrice": float(filter_item.max_price),
                    "tickSize": float(filter_item.tick_size)
                }
        
        return {
            "symbol": symbol_info.symbol,
            "status": symbol_info.status,
            "lot_size": lot_size,
            "min_notional": min_notional,
            "price_filter": price_filter
        }
        
    except Exception as e:
        print(f"Error getting exchange info: {e}")
        return None

async def get_24h_ticker(symbol="BTCEUR"):
    """Get 24h ticker statistics"""
    try:
        # Use the correct method name from the available methods
        response = client.rest_api.ticker24hr(symbol=symbol)
        ticker_data = response.data()
        
        # Access the actual instance
        actual_data = ticker_data.actual_instance if hasattr(ticker_data, 'actual_instance') else ticker_data
        
        return {
            "symbol": actual_data.symbol,
            "priceChange": float(actual_data.price_change),
            "priceChangePercent": float(actual_data.price_change_percent),
            "lastPrice": float(actual_data.last_price),
            "volume": float(actual_data.volume),
            "openPrice": float(actual_data.open_price),
            "highPrice": float(actual_data.high_price),
            "lowPrice": float(actual_data.low_price)
        }
        
    except Exception as e:
        print(f"Error getting 24h ticker: {e}")
        # If the attribute names are different, let's debug what's available
        try:
            response = client.rest_api.ticker24hr(symbol=symbol)
            ticker_data = response.data()
            actual_data = ticker_data.actual_instance if hasattr(ticker_data, 'actual_instance') else ticker_data
            
            print(f"Debug - Ticker data type: {type(actual_data)}")
            print(f"Debug - Ticker data: {actual_data}")
            if hasattr(actual_data, '__dict__'):
                print(f"Debug - Ticker attributes: {vars(actual_data)}")
        except Exception as debug_e:
            print(f"Debug failed: {debug_e}")
        return None

def round_to_step_size(quantity, step_size):
    """Round quantity down to match step_size precision"""
    step_decimal = Decimal(str(step_size))
    quantity_decimal = Decimal(str(quantity))
    
    rounded = float(quantity_decimal.quantize(step_decimal, rounding=ROUND_DOWN))
    return rounded

async def calculate_purchase_amount(symbol="BTCEUR", quote_amount=5.0):
    """Calculate purchase amount for DCA strategy"""
    try:
        exchange_info = await get_exchange_info(symbol)
        price_info = await get_current_price(symbol)
        
        if not exchange_info or not price_info:
            return None
        
        current_price = price_info["price"]
        lot_size = exchange_info["lot_size"]
        min_notional = exchange_info["min_notional"]
        
        if quote_amount < min_notional["minNotional"]:
            print(f"Error: Quote amount {quote_amount} is below minimum notional {min_notional['minNotional']}")
            return None
        
        raw_quantity = quote_amount / current_price
        step_size = lot_size["stepSize"]
        rounded_quantity = round_to_step_size(raw_quantity, step_size)
        
        if rounded_quantity < lot_size["minQty"]:
            print(f"Error: Rounded quantity {rounded_quantity} is below minimum quantity {lot_size['minQty']}")
            return None
        
        actual_cost = rounded_quantity * current_price
        
        return {
            "symbol": symbol,
            "quote_amount_requested": quote_amount,
            "current_price": current_price,
            "raw_quantity": raw_quantity,
            "rounded_quantity": rounded_quantity,
            "actual_cost": actual_cost,
            "step_size": step_size,
            "min_notional": min_notional["minNotional"],
            "lot_size": lot_size
        }
        
    except Exception as e:
        print(f"Error calculating purchase amount: {e}")
        return None

async def place_market_order(symbol="BTCEUR", quote_order_qty=5.0, dry_run=False):
    """Place a MARKET buy order using quote_order_qty"""
    try:
        if dry_run:
            print(f"[DRY RUN] Would place market order: {quote_order_qty} EUR for {symbol}")
            price_info = await get_current_price(symbol)
            if price_info:
                simulated_qty = quote_order_qty / price_info["price"]
                return {
                    "symbol": symbol,
                    "orderId": "DRY_RUN_12345",
                    "orderListId": -1,
                    "clientOrderId": "DRY_RUN",
                    "transactTime": int(datetime.now().timestamp() * 1000),
                    "price": "0.00000000",
                    "origQty": f"{simulated_qty:.8f}",
                    "executedQty": f"{simulated_qty:.8f}",
                    "cummulativeQuoteQty": f"{quote_order_qty:.8f}",
                    "status": "FILLED",
                    "timeInForce": "IOC",
                    "type": "MARKET",
                    "side": "BUY",
                    "workingTime": int(datetime.now().timestamp() * 1000),
                    "fills": [{
                        "price": f"{price_info['price']:.8f}",
                        "qty": f"{simulated_qty:.8f}",
                        "commission": "0.00000000",
                        "commissionAsset": symbol.split('EUR')[0],
                        "tradeId": 999999
                    }],
                    "dry_run": True
                }
            return None
        
        # Real order placement - let's try the actual API call
        try:
            print("Attempting to place real order...")
            response = client.rest_api.new_order(
                symbol=symbol,
                side="BUY",
                type="MARKET",
                quote_order_qty=quote_order_qty
            )
            
            print("✅ Order placed successfully!")
            order_data = response.data()
            
            # Handle if the response has actual_instance
            if hasattr(order_data, 'actual_instance'):
                return order_data.actual_instance
            else:
                return order_data
                
        except Exception as api_error:
            error_message = str(api_error)
            print(f"❌ API Error: {error_message}")
            
            if "Invalid API-key, IP, or permissions" in error_message:
                print("\n⚠️  API Permission Error Detected!")
                print("This appears to be an API key permission issue. Please check:")
                print("1. Your API key has 'Spot & Margin Trading' permissions enabled")
                print("2. Your IP address is whitelisted (if IP restriction is enabled)")
                print("3. Your API key and secret are correct")
                print("\nTo fix this:")
                print("1. Go to https://www.binance.com/en/my/settings/api-management")
                print("2. Edit your API key")
                print("3. Enable 'Spot & Margin Trading' permission")
                print("4. Add your current IP to the whitelist or disable IP restriction")
            elif "insufficient balance" in error_message.lower():
                print("\n💰 Insufficient Balance!")
                print("You don't have enough EUR in your account for this purchase.")
            elif "minimum notional" in error_message.lower():
                print("\n📏 Order too small!")
                print("The order amount is below the minimum required.")
            else:
                print(f"\n🔧 Debug Info:")
                print(f"Error type: {type(api_error)}")
                print(f"Error details: {api_error}")
            
            return None
        
    except Exception as e:
        print(f"Error placing market order: {e}")
        return None

async def verify_order_execution(order_result):
    """Verify and analyze order execution"""
    if not order_result:
        return None
    
    try:
        # Handle both object attributes and dictionary access
        def safe_get(obj, key, default=None):
            if hasattr(obj, key):
                return getattr(obj, key)
            elif hasattr(obj, 'get') and callable(getattr(obj, 'get')):
                return obj.get(key, default)
            elif isinstance(obj, dict):
                return obj.get(key, default)
            else:
                return default
        
        # Handle actual_instance if it exists
        if hasattr(order_result, 'actual_instance'):
            actual_data = order_result.actual_instance
        else:
            actual_data = order_result
        
        # Extract order data using safe_get
        executed_qty = float(safe_get(actual_data, "executedQty", 0) or safe_get(actual_data, "executed_qty", 0))
        cumulative_quote_qty = float(safe_get(actual_data, "cummulativeQuoteQty", 0) or safe_get(actual_data, "cumulative_quote_qty", 0))
        
        avg_price = 0.0
        total_commission = 0.0
        commission_asset = ""
        
        fills = safe_get(actual_data, "fills", [])
        if fills:
            total_value = 0.0
            total_qty = 0.0
            
            for fill in fills:
                # Handle fill data safely
                if hasattr(fill, 'price'):
                    fill_price = float(fill.price)
                    fill_qty = float(fill.qty)
                    fill_commission = float(fill.commission)
                    commission_asset = fill.commission_asset
                else:
                    fill_price = float(fill.get("price", 0))
                    fill_qty = float(fill.get("qty", 0))
                    fill_commission = float(fill.get("commission", 0))
                    commission_asset = fill.get("commissionAsset", "")
                
                total_value += fill_price * fill_qty
                total_qty += fill_qty
                total_commission += fill_commission
            
            if total_qty > 0:
                avg_price = total_value / total_qty
        
        execution_analysis = {
            "order_id": safe_get(actual_data, "orderId") or safe_get(actual_data, "order_id"),
            "symbol": safe_get(actual_data, "symbol"),
            "status": safe_get(actual_data, "status"),
            "side": safe_get(actual_data, "side"),
            "type": safe_get(actual_data, "type"),
            "executed_qty": executed_qty,
            "cumulative_quote_qty": cumulative_quote_qty,
            "average_price": avg_price,
            "total_commission": total_commission,
            "commission_asset": commission_asset,
            "number_of_fills": len(fills),
            "execution_time": safe_get(actual_data, "transactTime") or safe_get(actual_data, "transact_time"),
            "is_fully_filled": safe_get(actual_data, "status") == "FILLED",
            "dry_run": safe_get(actual_data, "dry_run", False)
        }
        
        return execution_analysis
        
    except Exception as e:
        print(f"Error verifying order execution: {e}")
        # Debug information
        print(f"Debug - Order result type: {type(order_result)}")
        print(f"Debug - Order result: {order_result}")
        if hasattr(order_result, '__dict__'):
            print(f"Debug - Order result attributes: {list(vars(order_result).keys())}")
        if hasattr(order_result, 'actual_instance'):
            actual = order_result.actual_instance
            print(f"Debug - Actual instance type: {type(actual)}")
            print(f"Debug - Actual instance: {actual}")
            if hasattr(actual, '__dict__'):
                print(f"Debug - Actual instance attributes: {list(vars(actual).keys())}")
        return None

async def safety_checks(symbol="BTCEUR", quote_amount=10.0):
    """Perform all safety checks before placing order"""
    try:
        print("🔍 Performing safety checks...")
        
        max_daily_spend = float(os.getenv("MAX_EUR_PER_DAY", "50.0"))
        max_price_change = float(os.getenv("MAX_PRICE_CHANGE_PERCENT", "12.0"))
        max_daily_orders = int(os.getenv("MAX_DAILY_ORDERS", "1"))
        
        # Check 1: 24h price volatility
        ticker_data = await get_24h_ticker(symbol)
        if not ticker_data:
            print("❌ Failed to get 24h ticker data")
            return False
        
        price_change_percent = abs(ticker_data["priceChangePercent"])
        if price_change_percent > max_price_change:
            print(f"❌ Price volatility too high: {price_change_percent:.2f}% > {max_price_change}%")
            return False
        
        print(f"✅ Price volatility check passed: {price_change_percent:.2f}%")
        
        # Check 2: Daily spending limit
        daily_stats = get_daily_stats()
        current_spent = daily_stats['total_spent']
        
        if current_spent + quote_amount > max_daily_spend:
            print(f"❌ Daily spending limit exceeded: {current_spent + quote_amount:.2f} > {max_daily_spend}")
            return False
        
        print(f"✅ Daily spending check passed: {current_spent + quote_amount:.2f} ≤ {max_daily_spend}")
        
        # Check 3: Daily order count
        daily_order_count = daily_stats['order_count']
        
        if daily_order_count >= max_daily_orders:
            print(f"❌ Daily order limit reached: {daily_order_count} ≥ {max_daily_orders}")
            return False
        
        print(f"✅ Daily order count check passed: {daily_order_count} < {max_daily_orders}")
        
        # Check 4: Market status
        exchange_info = await get_exchange_info(symbol)
        if not exchange_info or exchange_info['status'] != 'TRADING':
            print(f"❌ Market not trading: {exchange_info['status'] if exchange_info else 'Unknown'}")
            return False
        
        print(f"✅ Market status check passed: TRADING")
        print("✅ All safety checks passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in safety checks: {e}")
        return False

async def execute_dca_purchase(symbol="BTCEUR", quote_amount=5.0, dry_run=True):
    """Complete DCA purchase workflow"""
    try:
        print(f"\n--- Executing DCA Purchase ---")
        print(f"Symbol: {symbol}")
        print(f"Quote Amount: {quote_amount}")
        print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
        
        # Step 1: Calculate purchase amount (validation)
        purchase_info = await calculate_purchase_amount(symbol, quote_amount)
        if not purchase_info:
            print("Failed to calculate purchase amount")
            return None
        
        print(f"Current price: €{purchase_info['current_price']:.2f}")
        print(f"Expected quantity: ~{purchase_info['rounded_quantity']:.8f} {symbol.replace('EUR', '')}")
        
        # Step 2: Place market order
        order_result = await place_market_order(symbol, quote_amount, dry_run)
        if not order_result:
            print("Failed to place order")
            return None
        
        # Step 3: Verify execution
        execution_analysis = await verify_order_execution(order_result)
        if not execution_analysis:
            print("Failed to verify order execution")
            return None
        
        # Step 4: Display results
        print(f"\n--- Order Execution Results ---")
        print(f"Order ID: {execution_analysis['order_id']}")
        print(f"Status: {execution_analysis['status']}")
        print(f"Executed Quantity: {execution_analysis['executed_qty']:.8f} {symbol.replace('EUR', '')}")
        print(f"Total Cost: {execution_analysis['cumulative_quote_qty']:.2f} EUR")
        print(f"Average Price: €{execution_analysis['average_price']:.2f}")
        print(f"Commission: {execution_analysis['total_commission']:.8f} {execution_analysis['commission_asset']}")
        print(f"Number of Fills: {execution_analysis['number_of_fills']}")
        
        return {
            "purchase_info": purchase_info,
            "order_result": order_result,
            "execution_analysis": execution_analysis
        }
        
    except Exception as e:
        print(f"Error in DCA purchase workflow: {e}")
        return None

async def safe_dca_purchase(symbol="BTCEUR", quote_amount=10.0, dry_run=True):
    """DCA purchase with integrated safety checks and logging"""
    try:
        # Perform safety checks
        if not await safety_checks(symbol, quote_amount):
            return None
        
        # Execute DCA purchase
        result = await execute_dca_purchase(symbol, quote_amount, dry_run)
        
        if result and result['execution_analysis']:
            # Log to database
            log_dca_order(result['execution_analysis'], dry_run)
            
            # Send Discord alert
            await send_discord_alert(result['execution_analysis'], dry_run)
        
        return result
        
    except Exception as e:
        print(f"Error in safe DCA purchase: {e}")
        return None

class DCAScheduler:
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.is_running = False
        
    async def scheduled_dca_job(self):
        """Scheduled DCA job with safety checks"""
        try:
            print(f"\n=== Scheduled DCA Job Started at {datetime.now()} ===")
            
            symbol = os.getenv("DCA_SYMBOL", "BTCEUR")  # Changed default
            quote_amount = float(os.getenv("DCA_AMOUNT", "10.0"))
            dry_run = os.getenv("DCA_DRY_RUN", "true").lower() == "true"
            
            result = await safe_dca_purchase(symbol, quote_amount, dry_run)
            
            if result:
                print("✅ Scheduled DCA job completed successfully")
            else:
                print("❌ Scheduled DCA job failed")
                
        except Exception as e:
            print(f"Error in scheduled DCA job: {e}")
    
    def start_scheduler(self, schedule_time="09:00"):
        """Start the scheduler"""
        try:
            self.scheduler.add_job(
                self.scheduled_dca_job,
                CronTrigger(hour=int(schedule_time.split(':')[0]), 
                           minute=int(schedule_time.split(':')[1])),
                id='daily_dca',
                replace_existing=True
            )
            
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            self.scheduler.start()
            self.is_running = True
            print(f"✅ DCA Scheduler started - next run at {schedule_time}")
            
        except Exception as e:
            print(f"Error starting scheduler: {e}")
    
    def stop_scheduler(self):
        """Stop the scheduler"""
        if self.is_running:
            self.scheduler.shutdown(wait=False)
            self.is_running = False
            print("✅ DCA Scheduler stopped")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\nReceived signal {signum} - shutting down scheduler...")
        self.stop_scheduler()
        sys.exit(0)

async def test_exchange_info_debug():
    """Test function to debug exchange_info response"""
    try:
        print("Testing exchange_info...")
        response = client.rest_api.exchange_info(symbol="BTCEUR")  # Changed default
        print(f"Response type: {type(response)}")
        print(f"Response: {response}")
        
        data = response.data()
        print(f"Data type: {type(data)}")
        print(f"Data: {data}")
        
        if hasattr(data, 'actual_instance'):
            actual = data.actual_instance
            print(f"Actual instance type: {type(actual)}")
            print(f"Actual instance: {actual}")
            
            if hasattr(actual, 'symbols'):
                print(f"Has symbols: {len(actual.symbols)}")
                if actual.symbols:
                    symbol_info = actual.symbols[0]
                    print(f"Symbol info type: {type(symbol_info)}")
                    print(f"Symbol info: {symbol_info}")
                    
                    if hasattr(symbol_info, 'filters'):
                        print(f"Filters count: {len(symbol_info.filters)}")
                        for i, filter_item in enumerate(symbol_info.filters):
                            print(f"Filter {i}: {filter_item}")
                            print(f"Filter {i} type: {type(filter_item)}")
                            if hasattr(filter_item, '__dict__'):
                                print(f"Filter {i} attributes: {list(vars(filter_item).keys())}")
                            break  # Just show first filter
            else:
                print("No symbols attribute")
        else:
            print("No actual_instance attribute")
            
    except Exception as e:
        print(f"Test failed: {e}")

async def debug_order_response():
    """Debug function to understand order response structure"""
    try:
        print("=== Testing Order Response Structure ===")
        
        # Place a small dry run order to see the structure
        order_result = await place_market_order("BTCEUR", 5.0, dry_run=True)
        
        print(f"Order result type: {type(order_result)}")
        print(f"Order result: {order_result}")
        
        if hasattr(order_result, '__dict__'):
            print(f"Order result attributes: {list(vars(order_result).keys())}")
            for key, value in vars(order_result).items():
                print(f"  {key}: {type(value)} = {value}")
        
        if hasattr(order_result, 'actual_instance'):
            actual = order_result.actual_instance
            print(f"Actual instance type: {type(actual)}")
            print(f"Actual instance: {actual}")
            if hasattr(actual, '__dict__'):
                print(f"Actual instance attributes: {list(vars(actual).keys())}")
                
    except Exception as e:
        print(f"Debug failed: {e}")

# Add this to your main menu
async def main():
    """Main function with menu options"""
    try:
        # Initialize database
        init_database()
        
        print("=== DCA Bot Menu ===")
        print("1. Run DCA once (manual)")
        print("2. Start scheduler")
        print("3. View daily stats")
        print("4. Test connection")
        print("5. Debug exchange info")
        print("6. Debug order response")  # Add this line
        
        choice = input("Choose an option (1-6): ")  # Update this line
        
        if choice == "1":
            # Manual DCA execution
            symbol = input("Symbol (default: BTCEUR): ") or "BTCEUR"  # Changed default
            amount = float(input("Amount in EUR (default: 10.0): ") or "10.0")  # Changed currency
            dry_run = input("Dry run? (y/N): ").lower() == "y"
            
            result = await safe_dca_purchase(symbol, amount, dry_run)
            
        elif choice == "2":
            # Start scheduler
            schedule_time = input("Schedule time (HH:MM, default: 09:00): ") or "09:00"
            
            scheduler = DCAScheduler()
            scheduler.start_scheduler(schedule_time)
            
            print("Scheduler running. Press Ctrl+C to stop.")
            try:
                while scheduler.is_running:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                scheduler.stop_scheduler()
                
        elif choice == "3":
            # View daily stats
            stats = get_daily_stats()
            print(f"\n--- Daily Stats for {stats['date']} ---")
            print(f"Orders placed: {stats['order_count']}")
            print(f"Total spent: €{stats['total_spent']:.2f}")  # Changed currency symbol
            
        elif choice == "4":
            # Test connection
            price_info = await get_current_price("BTCEUR")  # Changed default
            if price_info:
                print(f"✅ Connection successful! BTC price: €{price_info['price']:.2f}")  # Changed currency symbol
            else:
                print("❌ Connection failed")
                
        elif choice == "5":
            # Debug exchange info
            await test_exchange_info_debug()
            
        elif choice == "6":  # Add this section
            # Debug order response
            await debug_order_response()
                
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    asyncio.run(main())