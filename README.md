# Trading Brain

Trading Brain adalah dashboard analisa trading yang menggabungkan chart, engine analisa, memory hasil signal, research, dan visual neuron dalam satu tempat.

Fokus project ini sederhana:
- membaca market multi-timeframe
- mencari setup yang rapi
- menjaga risiko
- belajar dari histori signal yang sudah lewat

## Yang Ada di Dalam

- `web/index.html`
  Dashboard utama.
- `brain_web.py`
  Server lokal untuk web dan API.
- `trading_brain/`
  Otak utama: engine, neuron, memory, training, recommendations, dan chat assistant.
- `visuals/`
  Viewer neuron 3D.
- `examples/`
  Contoh payload untuk tes cepat.
- `tests/`
  Test untuk logic utama.

## Fitur Utama

- analisa `crypto`, `forex`, `XAUUSD`, dan `WTI`
- baca timeframe `4H`, `1H`, `15M`, dan `5M`
- mode `balanced` dan `precision`
- `Best Setups` lintas pair
- `Signal Memory` untuk histori win/loss dan signal open
- `readiness meter` dan training report
- `Desk Chat` yang bisa jawab soal signal, setup, memory, readiness, dan news
- visual neuron yang sinkron dengan hasil analisa

## Jalanin Lokal

```powershell
python brain_web.py
```

Lalu buka:

`http://127.0.0.1:8765`

Kalau mau jalan dari contoh payload:

```powershell
python brain.py examples\bullish_btc.json
python brain.py --mode super examples\bullish_btc.json
```

## Test

```powershell
python -m unittest discover -s tests -v
```

## Publik

Versi web yang sedang dipakai:

`https://20.189.73.162/trading-brain/web/index.html`

## Catatan

Project ini masih terus berkembang, tapi fondasinya sudah siap dipakai:
- ada engine analisa
- ada memory database
- ada training adaptif
- ada web dashboard
- ada API

Kalau mau dipakai serius, langkah berikutnya yang paling masuk akal biasanya:
- sambung ke feed broker yang lebih kuat
- rapikan deployment production
- tambah monitoring dan logging yang lebih formal
