### Idée
Acheter automatiquement une toute petite quantité de BTC (ou autre) à intervalles réguliers (ex : 5 € par jour ou par semaine), avec des garde-fous (plafond de dépense, vérif du minNotional, pause si la volat augmente), et t’envoyer une alerte Telegram/Discord à chaque exécution. C’est parfait pour apprendre les ordres spot, la gestion des clés, et la robustesse d’un script — sans t’exposer à des leviers ou des futures.

### Stack & librairies
•	Python 3.11+
•	binance-connector (SDK officiel) ou ccxt (plus générique). Pour un débutant, je conseille binance-connector.
•	python-dotenv (charger tes clés depuis .env)
•	pydantic (valider la config)
•	requests (si tu envoies des webhooks / Telegram)
•	SQLite (sqlite3) ou CSV pour l’historique
•	APScheduler (si tu veux planifier “tous les jours à 09:00”) ou un simple cron

### Steps
## 1.	Initialisation du projet

•	Crée un venv, installe binance-connector, python-dotenv, pydantic, apscheduler.
•	Fichier .env : BINANCE_API_KEY, BINANCE_API_SECRET, SYMBOL=BTCUSDT, etc.

## 2.	Lecture des métadonnées & ticker

•	Endpoint GET /api/v3/exchangeInfo → récupère LOT_SIZE, MIN_NOTIONAL, PRICE_FILTER.
•	Endpoint GET /api/v3/ticker/price?symbol=BTCUSDT pour le prix courant.

## 3.	Calcul du montant à acheter

•	Mode simple : achète en quote (ex. 5 USDT) via quoteOrderQty.
•	Arrondis la quantité d’actif au stepSize si tu calcules en quantity.

## 4.	Passage d’un ordre MARKET spot

•	Endpoint POST /api/v3/order (type=MARKET, quoteOrderQty=…).
•	Vérifie la réponse (fills, cummulativeQuoteQty, etc.).

## 5.	Logs & alertes

•	Écris dans SQLite (date, symbol, quote spent, executedQty, price).
•	Envoie un message Telegram (Bot API) ou un webhook Discord avec le récap.

## 6.	Planification

•	Avec APScheduler (ou cron), programme l’exécution quotidienne/hebdo.
•	Ajoute un mode dry-run (ne pas appeler l’endpoint order, juste simuler).

## 7.	Garde-fous opérationnels

•	Avant de trader :
•	24h price change via GET /api/v3/ticker/24hr → si abs(priceChangePercent) > 12, skip.
•	Compteur d’ordres du jour ≤ 1.
•	Somme dépensée du jour ≤ MAX_EUR_PER_DAY.