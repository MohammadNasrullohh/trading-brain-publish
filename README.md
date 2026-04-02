# Trading Brain

Project ini sekarang punya brain trading yang benar-benar jalan sebagai kode Python, bukan cuma kumpulan Markdown.
Versi ini sudah di-upgrade menjadi brain multi-market dengan `137 neuron` yang sadar konteks `crypto`, `forex`, kualitas eksekusi, dan kondisi akun.
Di atas itu sekarang juga ada mode `SuperTradingAgent` yang mengubah output menjadi command center lengkap: mission posture, playbook aktif, desk consensus, scenario map, monitoring, dan risk protocol.

## File Utama

- `brain.py`
  - CLI untuk menjalankan analisa dari file JSON
- `trading_brain/`
  - package inti berisi engine, model data, dan neuron trading
- `examples/`
  - contoh input market untuk crypto bullish, crypto bearish, forex bullish, no-trade, dan pause mode saat akun sedang jelek
- `tests/`
  - test dasar untuk memastikan verdict engine stabil

## Cara Menjalankan

```powershell
python brain.py examples\bullish_btc.json
python brain.py examples\bearish_eth.json
python brain.py examples\forex_eurusd_bullish.json
python brain.py examples\no_trade_range.json
python brain.py examples\drawdown_pause.json
python brain.py --mode super examples\bullish_btc.json
python brain.py --mode super examples\drawdown_pause.json
python brain_web.py
```

Server web lokal default akan hidup di `http://127.0.0.1:8765` dan otomatis membuka viewer 3D yang memakai hasil analisa brain sebagai penggerak visual neuron.
Root URL sekarang diarahkan ke dashboard web utama di `web/index.html`, yang berisi:
- chart TradingView resmi
- JSON payload editor
- hasil analisa brain / super-agent
- iframe neuron brain yang sinkron ke analisa yang sama

## Cara Test

```powershell
python -m unittest discover -s tests -v
```

## Format Input

Brain menerima input JSON seperti ini:

```json
{
  "symbol": "BTCUSDT",
  "timeframe": "15m",
  "style": "intraday",
  "market_type": "crypto",
  "session": "us",
  "price": 68280,
  "open": 68100,
  "high": 68320,
  "low": 68050,
  "close": 68280,
  "atr": 180,
  "levels": {
    "support": [68120, 67880],
    "resistance": [68210, 68650, 68980]
  },
  "indicators": {
    "ema_fast": 68220,
    "ema_slow": 68160,
    "rsi": 58,
    "macd_histogram": 11,
    "volume_trend": "rising",
    "vwap": 68200,
    "adx": 28,
    "stochastic": 64,
    "open_interest_delta": 1.8,
    "funding_rate": 0.012
  },
  "risk": {
    "max_risk_percent": 0.5,
    "leverage": 3
  },
  "sentiment": {
    "score": 0.35,
    "headline_risk": false,
    "correlation_bias": 0.4,
    "macro_risk": false
  },
  "context": {
    "regime_hint": "trending",
    "session_quality_hint": "high"
  },
  "microstructure": {
    "spread": 6.0,
    "fee_bps": 4,
    "slippage_bps": 3,
    "weekend": false
  }
}
```

## Output

Brain akan mengeluarkan JSON berisi:
- summary
- context
- levels
- scores
- risk
- plan
- conditional_plan
- reasons
- warnings
- blockers

Mode `--mode super` akan mengeluarkan lapisan tambahan:
- `mission_control`
- `strategic_brief`
- `desk_consensus`
- `scenario_map`
- `action_plan`
- `monitoring`
- `risk_protocol`
- `brain_output`

Mode web sekarang memakai endpoint lokal:
- `GET /api/health`
- `GET /api/examples`
- `POST /api/analyze`

Viewer web di `visuals/neuron_brain_3d.html` sekarang bisa:
- fetch hasil analisa langsung dari engine Python
- menyalakan neuron berdasarkan `visual_state` dari brain
- fokus ke node penting yang dipilih brain
- tetap bisa dikontrol manual untuk rotate, pan, zoom, dan select

Dashboard `web/index.html` menambahkan:
- sinkronisasi payload ke chart TradingView
- pemilihan example dan brain mode dari browser
- panel readout verdict, confidence, reasons, warnings, blockers, dan plan snapshot

Ringkasan mode `--mode super` sekarang juga memuat:
- `readiness_score`
- `mission_posture`
- `dominant_playbook`

Context dan risk sekarang juga memuat:
- `market_type`
- `asset_profile`
- `session`
- `session_quality`
- `timeframe_minutes`
- `effective_risk_percent`
- `size_multiplier`
- `leverage_profile`
- `current_drawdown_percent`
- `account_heat`
- `loss_streak`
- `friction_bps`

## Catatan Tambahan

File Markdown lama tetap disimpan sebagai referensi desain prompt dan arsitektur neuron.
Viewer 3D di folder `visuals` sekarang mengikuti scene data dinamis dan sudah memetakan engine ratusan neuron ini.

Neuron level lanjut yang ditambahkan mencakup:
- compatibility antara style dan timeframe
- memori previous high/low dan session high/low
- orderbook pressure dan liquidity quality
- candle anatomy dan breakout quality
- trap detection dan execution location
- drawdown guard, recovery mode, dan account heat
- asymmetry check untuk reward-to-risk
- conviction calibration sebelum verdict akhir
- micro-neurons tambahan untuk quality scan, tactical planning, dan defensive veto

Lapisan `SuperTradingAgent` menambahkan perilaku seperti:
- memilih `mission_posture` seperti `PRESS`, `AMBUSH`, `SHIELD`, atau `LOCKDOWN`
- memilih `dominant_playbook` berbeda untuk crypto dan forex
- membuat `desk_consensus` antara macro, structure, execution, dan risk desk
- membuat `scenario_map` primary, contingency, dan abort case sebelum trade dieksekusi
