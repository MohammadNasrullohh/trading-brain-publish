# Trading Neurons

Folder ini berisi neuron modular untuk brain AI trading.

## Cara Baca

- `neuron01` sampai `neuron22` adalah modul instruksi.
- Setiap neuron punya fokus kerja sendiri.
- Kamu bisa gabungkan semua neuron untuk mode lengkap.
- Kamu juga bisa pilih beberapa neuron saja untuk mode ringan.

## Rekomendasi Susunan

Paket minimal:
- `neuron01` sampai `neuron07`
- `neuron16` sampai `neuron19`
- `neuron21`
- `neuron22`

Paket intraday:
- `neuron01` sampai `neuron10`
- `neuron11` sampai `neuron19`
- `neuron21`
- `neuron22`

Paket konservatif:
- `neuron01`
- `neuron02`
- `neuron04`
- `neuron05`
- `neuron06`
- `neuron07`
- `neuron15`
- `neuron16`
- `neuron17`
- `neuron18`
- `neuron19`
- `neuron21`
- `neuron22`

## Daftar Neuron

- `neuron01_persona_core.md` - identitas AI trading
- `neuron02_mission_guard.md` - misi dan batas perilaku
- `neuron03_input_reader.md` - membaca input user
- `neuron04_market_regime.md` - klasifikasi kondisi market
- `neuron05_higher_timeframe_bias.md` - bias timeframe besar
- `neuron06_structure_mapper.md` - market structure
- `neuron07_key_levels.md` - support, resistance, supply, demand
- `neuron08_liquidity_context.md` - sweep, trap, dan liquidity
- `neuron09_volume_momentum.md` - volume dan momentum
- `neuron10_volatility_engine.md` - volatility dan range
- `neuron11_long_setup.md` - validasi ide long
- `neuron12_short_setup.md` - validasi ide short
- `neuron13_breakout_retest.md` - skenario breakout-retest
- `neuron14_pullback_reversal.md` - skenario pullback dan reversal
- `neuron15_confluence_score.md` - skor kualitas setup
- `neuron16_risk_budget.md` - alokasi risiko
- `neuron17_stop_loss_architect.md` - desain stop loss
- `neuron18_target_engine.md` - take profit dan R:R
- `neuron19_no_trade_filter.md` - filter wait dan no-trade
- `neuron20_event_sentiment_guard.md` - filter news dan sentimen
- `neuron21_output_formatter.md` - format jawaban
- `neuron22_verdict_gate.md` - keputusan akhir
