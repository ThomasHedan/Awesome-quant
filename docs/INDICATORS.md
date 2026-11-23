# Bibliothèque d'indicateurs

180 indicateurs, calculés sur chaque timeframe (H1, H4, D1), plus 21 règles calendaires. Généré par `uv run fxgen indicators`.

Conditions générées par indicateur : `>pXX` (percentiles du train), `>0` (signé), `:rising`, `:cross_up` / `:cross_down` (croisement de 0), `>N` et `:cross_up@N` / `:cross_down@N` (niveaux). Les patterns binaires donnent `>0.5` (présent) et `:cross_up@0.5` (apparu dans les 5 dernières bougies).

## Tendance (62)

| Nom | Définition | Conditions en plus des percentiles |
|---|---|---|
| `dist_ema20` | (close - EMA20) / ATR14 | `>0`, croisement de 0 |
| `dist_ema50` | (close - EMA50) / ATR14 | `>0`, croisement de 0 |
| `dist_ema200` | (close - EMA200) / ATR14 | `>0`, croisement de 0 |
| `ema20_50` | (EMA20 - EMA50) / ATR14 | `>0`, croisement de 0 |
| `ema50_200` | (EMA50 - EMA200) / ATR14 | `>0`, croisement de 0 |
| `slope_ema50` | (EMA50 - EMA50[5]) / ATR14 | `>0` |
| `slope_ema200` | (EMA200 - EMA200[10]) / ATR14 | `>0` |
| `linreg_slope20` | LinReg slope(20) / ATR14 | `>0` |
| `adx14` | ADX(14) | niveaux 20, 25, 40 |
| `di_diff14` | +DI(14) - -DI(14) | `>0`, croisement de 0 |
| `macd_hist` | MACD(12,26,9) histogram / ATR14 | `>0`, croisement de 0 |
| `macd_line` | MACD(12,26) line / ATR14 | `>0` |
| `aroon_osc25` | Aroon oscillator(25) | `>0`, croisement de 0, niveaux -50, 50 |
| `efficiency20` | Kaufman efficiency ratio(20) | - |
| `dist_sma20` | (close - SMA20) / ATR14 | `>0`, croisement de 0 |
| `dist_sma50` | (close - SMA50) / ATR14 | `>0`, croisement de 0 |
| `dist_sma100` | (close - SMA100) / ATR14 | `>0`, croisement de 0 |
| `dist_sma200` | (close - SMA200) / ATR14 | `>0`, croisement de 0 |
| `dist_ema9` | (close - EMA9) / ATR14 | `>0`, croisement de 0 |
| `dist_ema100` | (close - EMA100) / ATR14 | `>0`, croisement de 0 |
| `dist_wma20` | (close - WMA20) / ATR14 | `>0`, croisement de 0 |
| `dist_hma9` | (close - HMA9) / ATR14 | `>0`, croisement de 0 |
| `dist_hma21` | (close - HMA21) / ATR14 | `>0`, croisement de 0 |
| `dist_hma55` | (close - HMA55) / ATR14 | `>0`, croisement de 0 |
| `dist_hma100` | (close - HMA100) / ATR14 | `>0`, croisement de 0 |
| `dist_alma9` | (close - ALMA9) / ATR14 | `>0`, croisement de 0 |
| `dist_alma50` | (close - ALMA50) / ATR14 | `>0`, croisement de 0 |
| `dist_dema20` | (close - DEMA20) / ATR14 | `>0`, croisement de 0 |
| `dist_tema20` | (close - TEMA20) / ATR14 | `>0`, croisement de 0 |
| `dist_kama10` | (close - KAMA10) / ATR14 | `>0`, croisement de 0 |
| `ema9_21` | (EMA9 - EMA21) / ATR14 | `>0`, croisement de 0 |
| `ema12_26` | (EMA12 - EMA26) / ATR14 | `>0`, croisement de 0 |
| `ema21_55` | (EMA21 - EMA55) / ATR14 | `>0`, croisement de 0 |
| `sma10_30` | (SMA10 - SMA30) / ATR14 | `>0`, croisement de 0 |
| `sma20_50` | (SMA20 - SMA50) / ATR14 | `>0`, croisement de 0 |
| `sma50_200` | (SMA50 - SMA200) / ATR14 | `>0`, croisement de 0 |
| `hma9_21` | (HMA9 - HMA21) / ATR14 | `>0`, croisement de 0 |
| `hma21_55` | (HMA21 - HMA55) / ATR14 | `>0`, croisement de 0 |
| `slope_ema20` | (EMA20 - EMA20[3]) / ATR14 | `>0`, croisement de 0 |
| `slope_sma50` | (SMA50 - SMA50[5]) / ATR14 | `>0`, croisement de 0 |
| `slope_hma9` | (HMA9 - HMA9[1]) / ATR14 | `>0`, croisement de 0 |
| `slope_hma21` | (HMA21 - HMA21[1]) / ATR14 | `>0`, croisement de 0 |
| `slope_hma55` | (HMA55 - HMA55[1]) / ATR14 | `>0`, croisement de 0 |
| `linreg_slope50` | LinReg slope(50) / ATR14 | `>0` |
| `linreg_slope100` | LinReg slope(100) / ATR14 | `>0` |
| `efficiency10` | Kaufman efficiency ratio(10) | - |
| `efficiency50` | Kaufman efficiency ratio(50) | - |
| `adx7` | ADX(7) | niveaux 20, 25, 40 |
| `di_diff7` | +DI(7) - -DI(7) | `>0`, croisement de 0 |
| `aroon_osc14` | Aroon oscillator(14) | `>0`, croisement de 0, niveaux -50, 50 |
| `adx14_slope` | ADX(14) - ADX(14)[5] | `>0` |
| `dist_supertrend10_3` | (close - SuperTrend(10, 3)) / ATR14 | `>0`, croisement de 0 |
| `dist_supertrend20_2` | (close - SuperTrend(20, 2)) / ATR14 | `>0`, croisement de 0 |
| `dist_psar` | (close - Parabolic SAR(0.02, 0.02, 0.2)) / ATR14 | `>0`, croisement de 0 |
| `ichimoku_tk` | (Tenkan - Kijun) / ATR14 | `>0`, croisement de 0 |
| `dist_kijun` | (close - Kijun) / ATR14 | `>0`, croisement de 0 |
| `dist_cloud` | (close - Ichimoku cloud middle) / ATR14 | `>0`, croisement de 0 |
| `cloud_thickness` | (Senkou A - Senkou B) / ATR14 (bullish cloud > 0) | `>0`, croisement de 0 |
| `vortex14` | Vortex VI+ - VI- (14) | `>0`, croisement de 0 |
| `trix15` | TRIX(15) of log close (x10000) | `>0`, croisement de 0 |
| `dpo20` | Detrended price oscillator(20) / ATR14 (non-centred) | `>0`, croisement de 0 |
| `ema_stack` | EMA alignment score: sign(EMA9-EMA21) + sign(EMA21-EMA50) + sign(EMA50-EMA200) | `>0`, niveaux -2.5, 2.5 |

## Momentum (40)

| Nom | Définition | Conditions en plus des percentiles |
|---|---|---|
| `rsi2` | RSI(2) | niveaux 10, 90 |
| `rsi7` | RSI(7) | niveaux 30, 50, 70 |
| `rsi14` | RSI(14) | niveaux 30, 50, 70 |
| `stoch5` | Stochastic %K(5) | niveaux 20, 80 |
| `stoch14` | Stochastic %K(14) | niveaux 20, 80 |
| `cci20` | CCI(20) | `>0`, croisement de 0, niveaux -100, 100 |
| `roc1_atr` | (close - close[1]) / ATR14 | `>0` |
| `roc5_atr` | (close - close[5]) / ATR14 | `>0` |
| `roc20_atr` | (close - close[20]) / ATR14 | `>0`, croisement de 0 |
| `roc60_atr` | (close - close[60]) / ATR14 | `>0` |
| `roc20_rank` | percentrank(ROC(20), 250) | - |
| `rsi21` | RSI(21) | niveaux 30, 50, 70 |
| `roc10_atr` | (close - close[10]) / ATR14 | `>0` |
| `roc5_rank` | percentrank(ROC(5), 250) | - |
| `roc60_rank` | percentrank(ROC(60), 250) | - |
| `stoch_slow14` | Slow stochastic %K(14,3) | niveaux 20, 50, 80 |
| `stoch_slow5` | Slow stochastic %K(5,3) | niveaux 20, 50, 80 |
| `stoch_slow21` | Slow stochastic %K(21,5) | niveaux 20, 50, 80 |
| `stoch_kd14` | Slow stochastic %K - %D (14,3,3) | `>0`, croisement de 0 |
| `stoch_kd5` | Slow stochastic %K - %D (5,3,3) | `>0`, croisement de 0 |
| `cci50` | CCI(50) | `>0`, croisement de 0, niveaux -100, 100 |
| `macd_hist_5_35_5` | MACD(5,35,5) histogram / ATR14 | `>0`, croisement de 0 |
| `macd_hist_8_17_9` | MACD(8,17,9) histogram / ATR14 | `>0`, croisement de 0 |
| `macd_hist_19_39_9` | MACD(19,39,9) histogram / ATR14 | `>0`, croisement de 0 |
| `macd_line_5_35` | MACD(5,35) line / ATR14 | `>0`, croisement de 0 |
| `macd_line_8_17` | MACD(8,17) line / ATR14 | `>0`, croisement de 0 |
| `stoch_rsi` | Stochastic RSI %K(14,14,3) | niveaux 20, 50, 80 |
| `stoch_rsi_kd` | Stochastic RSI %K - %D | `>0`, croisement de 0 |
| `rsi_signal` | RSI(14) - SMA(RSI(14), 9) | `>0`, croisement de 0 |
| `williams_r14` | Williams %R(14) | niveaux -80, -20 |
| `tsi` | True strength index(13, 25) | `>0`, croisement de 0, niveaux -0.25, 0.25 |
| `tsi_signal` | TSI - EMA(TSI, 7) | `>0`, croisement de 0 |
| `cmo14` | Chande momentum oscillator(14) | `>0`, croisement de 0, niveaux -50, 50 |
| `ultimate` | Ultimate oscillator(7, 14, 28) | niveaux 30, 50, 70 |
| `fisher10` | Fisher transform(10) of hl2 | `>0`, croisement de 0, niveaux -1.5, 1.5 |
| `coppock` | Coppock curve (WMA10 of ROC14 + ROC11) | `>0`, croisement de 0 |
| `awesome` | Awesome oscillator / ATR14 | `>0`, croisement de 0 |
| `elder_bull` | Elder bull power (high - EMA13) / ATR14 | `>0`, croisement de 0 |
| `elder_bear` | Elder bear power (low - EMA13) / ATR14 | `>0`, croisement de 0 |
| `momentum_rank` | percentrank(close - close[10], 100) | niveaux 10, 90 |

## Volatilité (20)

| Nom | Définition | Conditions en plus des percentiles |
|---|---|---|
| `atr_rank` | percentrank(ATR14, 250) | - |
| `atr_ratio` | ATR5 / ATR50 | - |
| `bb_width_rank` | percentrank(Bollinger width(20,2), 250) | - |
| `bb_pctb` | Bollinger %B(20,2) | niveaux 0, 0.5, 1 |
| `range_atr` | (high - low) / ATR14 | - |
| `ret_vol_ratio` | stdev(log ret, 10) / stdev(log ret, 100) | - |
| `chop14` | Choppiness index(14) | niveaux 38.2, 61.8 |
| `atr_z` | zscore(ATR14, 250) | `>0` |
| `atr_rank50` | percentrank(ATR14, 50) | - |
| `atr_ratio14_100` | ATR14 / ATR100 | niveaux 1 |
| `hv_rank` | percentrank(stdev(log ret, 20), 250) | - |
| `bb_pctb50` | Bollinger %B(50,2) | niveaux 0, 0.5, 1 |
| `bb_width_z` | zscore(Bollinger width(20,2), 100) | `>0` |
| `squeeze` | Bollinger(20,2) width / Keltner(20,1.5 ATR) width (< 1 = squeeze) | niveaux 1 |
| `keltner_pos` | (close - EMA20) / (2 x ATR10) (Keltner position) | `>0`, niveaux -1, 1 |
| `ulcer14` | Ulcer index(14) (%) | - |
| `mass_index` | Mass index(9, 25) | niveaux 26.5, 27 |
| `chop50` | Choppiness index(50) | niveaux 38.2, 61.8 |
| `range_rank` | percentrank(high - low, 100) | niveaux 10, 90 |
| `range_ratio5_20` | mean range 5 / mean range 20 | niveaux 1 |

## Structure de prix et patterns (50)

| Nom | Définition | Conditions en plus des percentiles |
|---|---|---|
| `donchian_pos20` | position in Donchian(20) channel | - |
| `donchian_pos55` | position in Donchian(55) channel | - |
| `dist_high20` | (highest high 20 - close) / ATR14 | - |
| `dist_low20` | (close - lowest low 20) / ATR14 | - |
| `dist_high100` | (highest high 100 - close) / ATR14 | - |
| `dist_low100` | (close - lowest low 100) / ATR14 | - |
| `bars_since_high20` | bars since 20-bar high | - |
| `bars_since_low20` | bars since 20-bar low | - |
| `close_location` | (close - low) / (high - low) | - |
| `body_ratio` | (close - open) / (high - low) | `>0` |
| `upper_wick` | upper wick / range | - |
| `lower_wick` | lower wick / range | - |
| `prev_bar_return` | (close[1] - open[1]) / ATR14 | `>0` |
| `donchian_pos10` | position in Donchian(10) channel | - |
| `donchian_pos100` | position in Donchian(100) channel | - |
| `dist_high10` | (highest high 10 - close) / ATR14 | - |
| `dist_low10` | (close - lowest low 10) / ATR14 | - |
| `dist_high50` | (highest high 50 - close) / ATR14 | - |
| `dist_low50` | (close - lowest low 50) / ATR14 | - |
| `dist_high250` | (highest high 250 - close) / ATR14 | - |
| `dist_low250` | (close - lowest low 250) / ATR14 | - |
| `bars_since_high50` | bars since 50-bar high | - |
| `bars_since_low50` | bars since 50-bar low | - |
| `bars_since_high100` | bars since 100-bar high | - |
| `bars_since_low100` | bars since 100-bar low | - |
| `close_range_pos20` | position of close in its 20-bar closing range | niveaux 0.5 |
| `close_range_pos100` | position of close in its 100-bar closing range | niveaux 0.5 |
| `streak` | consecutive up (+) / down (-) closes | `>0`, niveaux -3, 3 |
| `dist_donchian_mid20` | (close - Donchian(20) middle) / ATR14 | `>0`, croisement de 0 |
| `breakout_up20` | close above the previous 20-bar high (in ATR) | `>0`, croisement de 0 |
| `breakout_down20` | close below the previous 20-bar low (in ATR, positive = below) | `>0`, croisement de 0 |
| `breakout_up55` | close above the previous 55-bar high (in ATR) | `>0`, croisement de 0 |
| `breakout_down55` | close below the previous 55-bar low (in ATR, positive = below) | `>0`, croisement de 0 |
| `body_atr` | (close - open) / ATR14 | `>0` |
| `body_sum5` | sum of the last 5 bodies / ATR14 | `>0`, croisement de 0 |
| `up_bars10` | share of up closes over 10 bars | niveaux 0.3, 0.7 |
| `inside_bar` | inside bar | pattern 0/1 (pas de percentiles) |
| `outside_bar` | outside bar | pattern 0/1 (pas de percentiles) |
| `higher_high` | higher high and higher low | pattern 0/1 (pas de percentiles) |
| `lower_low` | lower high and lower low | pattern 0/1 (pas de percentiles) |
| `bull_engulfing` | bullish engulfing | pattern 0/1 (pas de percentiles) |
| `bear_engulfing` | bearish engulfing | pattern 0/1 (pas de percentiles) |
| `bull_pinbar` | bullish pin bar (lower wick > 60% of range) | pattern 0/1 (pas de percentiles) |
| `bear_pinbar` | bearish pin bar (upper wick > 60% of range) | pattern 0/1 (pas de percentiles) |
| `doji` | doji (body < 10% of range) | pattern 0/1 (pas de percentiles) |
| `marubozu_up` | bullish marubozu (body > 85% of range) | pattern 0/1 (pas de percentiles) |
| `marubozu_down` | bearish marubozu (body > 85% of range) | pattern 0/1 (pas de percentiles) |
| `three_up` | three higher closes | pattern 0/1 (pas de percentiles) |
| `three_down` | three lower closes | pattern 0/1 (pas de percentiles) |
| `nr7` | narrowest range of the last 7 bars | pattern 0/1 (pas de percentiles) |

## Volume (ignorés sans volume) (8)

| Nom | Définition | Conditions en plus des percentiles |
|---|---|---|
| `volume_z` | zscore(volume, 50) | `>0` |
| `obv_flow10` | 10-bar OBV change / 10-bar volume | `>0` |
| `mfi14` | Money flow index(14) | niveaux 20, 80 |
| `volume_rank` | percentrank(volume, 100) | niveaux 10, 90 |
| `volume_ratio` | SMA(volume, 5) / SMA(volume, 50) | niveaux 1 |
| `cmf20` | Chaikin money flow(20) | `>0`, croisement de 0 |
| `dist_vwma20` | (close - VWMA20) / ATR14 | `>0`, croisement de 0 |
| `force13` | EMA13(change x volume) / (SMA13(volume) x ATR14) | `>0`, croisement de 0 |

## Calendrier (21)

| Nom | Définition |
|---|---|
| `session_asia` | decision time in asia session (22-07 UTC) |
| `session_london` | decision time in london session (07-16 UTC) |
| `session_newyork` | decision time in newyork session (12-21 UTC) |
| `session_overlap` | decision time in overlap session (12-16 UTC) |
| `day_monday` | decision day is Monday (UTC) |
| `day_tuesday` | decision day is Tuesday (UTC) |
| `day_wednesday` | decision day is Wednesday (UTC) |
| `day_thursday` | decision day is Thursday (UTC) |
| `day_friday` | decision day is Friday (UTC) |
| `hour_asia_early` | decision time early Asia (22-02 UTC) |
| `hour_tokyo` | decision time Tokyo (00-06 UTC) |
| `hour_london_open` | decision time London open (07-10 UTC) |
| `hour_london_midday` | decision time London midday (10-13 UTC) |
| `hour_ny_open` | decision time New York open (13-16 UTC) |
| `hour_ny_afternoon` | decision time New York afternoon (16-19 UTC) |
| `hour_late_ny` | decision time late New York (19-22 UTC) |
| `day_monday_am` | Monday before 10 UTC |
| `day_friday_pm` | Friday from 14 UTC |
| `month_start` | first 3 days of the month |
| `month_mid` | days 13-17 of the month |
| `month_end` | from the 27th of the month |
