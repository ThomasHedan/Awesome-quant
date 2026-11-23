# fxgen — générateur génétique d'arbres de décision pour le swing trading FX

fxgen génère des stratégies FX sous forme de petits arbres de décision (profondeur 3–4), les fait évoluer
avec un algorithme génétique, puis **ne garde que celles qui survivent à une batterie de tests de robustesse
stricte**. La priorité est d'éviter le data-mining bias et le sur-apprentissage, pas d'afficher de beaux
backtests. Le pipeline est pilotable depuis Claude via un serveur MCP.

```
CSV OHLCV ─► validation ─► features (180 indicateurs × H1/H4/D1, normalisés)
          ─► ~7600 conditions binaires (seuils = percentiles du TRAIN, niveaux, croisements) ─► filtre de redondance
          ─► GA sur le TRAIN uniquement ─► candidats diversifiés
          ─► validation + walk-forward + permutation + DSR + sensibilité + Monte Carlo
          ─► finalistes ─► HOLDOUT (une seule fois) ─► classement, rapports HTML, Pine Script v6
```

## Installation

Python 3.11+ et [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run pytest            # ~100 tests, ~15 min
```

## Démarrage rapide sur données synthétiques

```bash
uv run fxgen synth                                   # 12 paires, H1/H4/D1, 2015-2024 -> data/synthetic/
uv run fxgen --config config/demo.yaml evolve        # GA + validation + robustesse (~10 min, 4 cœurs)
uv run fxgen --config config/demo.yaml holdout       # holdout des finalistes, une seule fois
```

Les sorties se trouvent dans `output/demo/<run_id>/` : `ranking.csv`, `ranking.json`, `reports/<tree>.html`
et `pine/<tree>.pine`.

Le jeu synthétique contient un **edge de momentum faible et volontaire** (`--edge 0.02`, à l'horizon de
5 jours), que le pipeline doit retrouver. Avec `uv run fxgen synth --edge 0`, les données sont du bruit pur
et rien ne doit survivre à la validation (`tests/test_validation.py::test_nothing_survives_on_pure_noise`).

`config/demo.yaml` réduit les budgets pour aller vite. Pour un vrai travail, utilisez `config/default.yaml`
(19 permutations, GA 200 × 30 : environ 1 h 15 estimée sur 4 cœurs pour 12 paires en H1 sur 10 ans).

## Données réelles gratuites : London Strategic Edge

[London Strategic Edge](https://londonstrategicedge.com/data) fournit gratuitement l'historique de 60 paires
forex via une API, avec une clé gratuite (page Databank). fxgen sait le télécharger :

```bash
export LSE_API_KEY=ta_cle                       # ne la committe jamais
uv run fxgen lse-download --estimate            # lignes que coûtera chaque paire (ne consomme pas de quota)
uv run fxgen lse-download                       # 12 paires, 10 dernières années, H1 -> data/lse/
uv run fxgen --config config/lse.yaml datasets  # contrôle qualité
uv run fxgen --config config/lse.yaml evolve
```

- Seul le H1 est téléchargé ; H4 et D1 sont reconstruits localement (bascule journalière à 22 h UTC).
- **Limites du plan gratuit** : un export lit au plus 2,5 M de lignes, et un an de H1 forex en lit environ
  375 000, car les bougies sont construites à partir des barres 1 minute. Le quota est de 12,5 M de lignes par
  jour, soit environ 3 paires × 10 ans par jour. Pour 12 paires, comptez donc environ 4 jours.
- **Reprise** : quand le quota est épuisé, la commande s'arrête proprement. Relancez-la le lendemain : les
  tranches déjà téléchargées (dans `data/lse/raw/`, vérifiées par SHA-256) ne sont pas redemandées.
- **Fenêtre et splits** : la fenêtre gratuite est de 10 ans. `config/lse.yaml` garde donc la première année
  comme période de chauffe des indicateurs : train 2018-2022, validation 2023-2024, holdout depuis 2025.
- Les conditions d'utilisation du site autorisent l'accès automatisé uniquement via leur API. Elles
  interdisent de partager la clé ou de redistribuer les données comme service concurrent.

## Brancher vos données

Un fichier CSV par paire et par timeframe, nommé selon `data.file_pattern` (`{pair}_{timeframe}.csv` par
défaut). Les colonnes se configurent dans le YAML :

```yaml
data:
  directory: /chemin/vers/mes/csv
  file_pattern: "{pair}_{timeframe}.csv"
  columns: {time: Date, open: Open, high: High, low: Low, close: Close, volume: Volume}   # volume: null si absent
  source_timezone: Europe/Paris      # converti en UTC au chargement
  timestamp_label: open              # "open" ou "close" : ce que représente l'horodatage d'une bougie
  pairs: [EURUSD, GBPUSD, ...]
  execution_timeframe: H1
  context_timeframes: [H4, D1]
```

`uv run fxgen datasets` affiche, pour chaque fichier, les plages de dates, les trous hors week-end, les
doublons retirés et les bougies OHLC réparées.

## Configuration (`config/default.yaml`)

| Section | Contenu |
|---|---|
| `splits` | `train_start` < `validation_start` < `holdout_start` < `end` |
| `conditions` | percentiles, seuil de redondance (\|corr\| > 0.9), taux d'activation min/max, filtres (familles, indicateurs exclus, conditions incluses / exclues) |
| `tree` | profondeur max (3–4) |
| `backtest` | stop / TP en multiples d'ATR (`atr_timeframe: D1` pour du swing), sortie temporelle, sortie au changement de branche, risque par trade |
| `costs` | spread par paire (pips), commission aller-retour, swap long/short par nuit (triple le mercredi) |
| `ga` | population, générations, tournoi, crossover, mutation, élitisme, rayon de partage (niching), `n_jobs` |
| `fitness` | trades min (total et par paire), trades max par paire et par an, pénalité de complexité, dépendance à une paire ou une période |
| `validation` | walk-forward, permutation, sensibilité, Monte Carlo, `dsr_min`, nombre de candidats et de finalistes |
| `prop_firm` | DD max, perte journalière max, risque minimum exigé par le test Monte Carlo |

Toute clé peut être surchargée en ligne de commande : `uv run fxgen --set ga.generations=50 --set tree.max_depth=4 evolve`.

## Garanties anti-biais

- **Pas de lookahead.** Les signaux sont calculés à la clôture de la bougie `t` et exécutés à l'ouverture de
  `t+1`. Une bougie H4 ou D1 n'est visible qu'une fois clôturée (`data/mtf.py`). Des tests corrompent toutes
  les données futures et vérifient que rien ne change dans le passé : features, conditions, alignement et
  trades.
- **Seuils sur le train uniquement.** Les percentiles sont calculés sur le train, toutes paires confondues,
  à partir de features normalisées (ratio à l'ATR, oscillateurs bornés, rangs percentiles et z-scores
  glissants).
- **Le GA ne voit que le train.** `evolve` refuse tout dataset qui chevauche le holdout.
- **Holdout verrouillé.** Il n'est lu qu'une fois par run, sur les finalistes (`fxgen holdout`), et une
  seconde exécution est refusée. Chaque `backtest_tree` manuel sur le holdout est journalisé
  (`holdout_access`) et augmente le nombre d'essais du DSR.
- **Deflated Sharpe Ratio** (Bailey & López de Prado). N = tous les arbres évalués par les GA, plus chaque
  backtest manuel, plus chaque accès au holdout. La variance de référence est celle de l'estimateur de Sharpe
  sous l'hypothèse nulle (1/(T−1)). La variance empirique d'une population GA est dominée par des perdants
  « déterministes » (coûts) et rendrait le test inutilisable. Le critère éliminatoire est la DSR sur le
  train, là où a eu lieu la sélection. Elle est aussi reportée sur la validation et le holdout.
- **Permutation.** On mélange l'ordre des bougies de chaque paire (forme relative conservée), on
  reconstruit H4 et D1, puis on relance tout le pipeline (features, seuils, redondance, GA) avec le même
  budget. Avec n permutations, la p-value minimale est 1/(n+1), d'où 19 runs par défaut.
- **Batterie par arbre :**
  - validation positive ;
  - DSR train ≥ `dsr_min` ;
  - walk-forward (fenêtres OOS majoritairement positives) ;
  - sensibilité (percentiles ±5, sorties ×0.8 / ×1.2, Sharpe ≥ 50 % de la base) ;
  - Monte Carlo sur l'ordre des trades (DD à 95 % compatible avec le DD max de la prop firm à au moins
    `min_risk_per_trade_pct` de risque) ;
  - perte journalière : la pire journée réalisée doit rester sous `max_daily_loss_pct` au même risque. Le
    rapport indique aussi le nombre max de positions simultanées et le risque compatible si toutes étaient
    stoppées le même jour ;
  - permutation.
- **Walk-forward du processus.** Le GA est relancé sur chaque fenêtre IS et son meilleur arbre est testé
  OOS. Le résultat (efficiency) est stocké avec le run.

## Bibliothèque d'indicateurs

180 indicateurs, chacun calculé en H1, H4 et D1 (540 séries), plus 21 règles calendaires. La liste complète
est dans [`docs/INDICATORS.md`](docs/INDICATORS.md) (régénérée par `uv run fxgen indicators`).

| Famille | Nombre | Exemples |
|---|---|---|
| Tendance | 62 | distances aux SMA/EMA/WMA/HMA/ALMA/DEMA/TEMA/KAMA, croisements de MA (9/21, 20/50, golden cross 50/200, HMA 9/21…), pentes, MACD, ADX/DI, Aroon, SuperTrend, Parabolic SAR, Ichimoku (TK, Kijun, nuage), Vortex, TRIX, DPO |
| Momentum | 40 | RSI 2/7/14/21, stochastiques rapide / lente / RSI et leurs croisements %K/%D, MACD (5,35,5), (8,17,9), (19,39,9), CCI, Williams %R, TSI, CMO, Ultimate, Fisher, Coppock, Awesome, Elder Ray, ROC |
| Volatilité | 20 | rangs d'ATR, ratios d'ATR, largeur et %B de Bollinger, squeeze Bollinger/Keltner, position Keltner, Ulcer, Mass index, Choppiness |
| Structure et patterns | 50 | Donchian, distances aux plus hauts / plus bas (10 à 250), cassures 20/55, streak, inside / outside bar, engulfing, pin bar, doji, marubozu, NR7 |
| Volume | 8 | z-score et rang du volume, OBV, MFI, CMF, VWMA, Force index |
| Calendrier | 21 | sessions, ouvertures Londres / New York, jours, lundi matin, vendredi après-midi, début / milieu / fin de mois |

Chaque indicateur produit plusieurs conditions :
- au-dessus de 11 percentiles du train (5, 10, 20… 90, 95) ;
- au-dessus de 0 pour les séries signées ;
- en hausse ;
- croisement de 0 ;
- au-dessus de ses niveaux classiques (RSI 30/50/70, stochastiques 20/80, CCI ±100, ADX 20/25/40…) et
  croisement de ces niveaux dans les 5 dernières bougies.

Les patterns de bougies donnent « présent » et « apparu dans les 5 dernières bougies ». Cela fait environ
7 600 conditions candidates ; le filtre de redondance en garde environ 6 100.

Pour trier, sans toucher au code :

```yaml
conditions:
  families: [trend, momentum]              # null = toutes
  exclude_features: ["stoch5", "*_rank"]   # indicateurs non calculés
  include_conditions: ["D1.*", "H4.*"]     # par exemple : uniquement le contexte H4 / D1
  exclude_conditions: ["*:cross_*@*"]      # par exemple : sans les croisements de niveaux
```

**Mémoire** : les conditions sont évaluées à la volée à partir des indicateurs, donc la mémoire ne dépend
pas du nombre de seuils. Pour 12 paires en H1 sur 10 ans, comptez environ 3 Go en régime établi et 4,5 Go
au pic. Les filtres ci-dessus réduisent ces chiffres.

## Type de stratégie et portefeuille décorrélé

### Identification du type

Chaque candidat reçoit une étiquette : `trend following`, `momentum`, `breakout`, `mean reversion` ou
`mixed`. Elle apparaît dans le classement, dans `get_rankings` / `get_tree` et via `fxgen classify TREE_ID`
ou l'outil MCP `classify_strategy`. Deux lectures sont combinées :

- **Structure de l'arbre.** Chaque condition sur le chemin d'une feuille LONG / SHORT est lue « dans le sens
  du mouvement » ou « contre lui ». Chaque indicateur a un style (tendance, momentum, cassure, filtre) et une
  polarité. Acheter quand `dist_ema50 > p70` est du suivi de tendance ; acheter quand `rsi14 > p20` est faux
  (RSI très bas) est du retour à la moyenne. Volatilité, calendrier et patterns neutres sont des filtres,
  indiqués en tags.
- **Comportement des trades.** À chaque entrée, le trade va-t-il dans le sens du mouvement de la veille et des
  5 derniers jours (`move_1d`, `move_5d`) ? Le profil de gains (taux de réussite, gain moyen / perte moyenne)
  est aussi donné.

La confiance augmente quand les deux lectures concordent ; un désaccord est signalé.

### Agrégateur de stratégies non corrélées

`fxgen portfolio`, ou l'outil MCP `build_portfolio`, assemble un portefeuille :

1. il part des arbres ayant survécu à la batterie, sur un ou plusieurs runs ;
2. il les trie par DSR de validation ;
3. il les ajoute un par un tant que la corrélation des rendements journaliers (train + validation) avec chaque
   membre reste sous `max_correlation`, avec au plus `max_per_type` stratégies du même type ;
4. il calcule les pondérations (égales ou inverses de la volatilité), les métriques par split, la corrélation
   moyenne, le ratio de diversification, et un Monte Carlo sur les journées du portefeuille ;
5. il en déduit le risque maximal par trade et par stratégie compatible avec le drawdown max et la perte
   journalière de la prop firm.

Le rapport HTML est écrit dans `output/portfolios/`. La sélection n'utilise que le train et la validation.
Les chiffres holdout n'apparaissent que si chaque membre a déjà été évalué par l'étape holdout de son run :
construire un portefeuille ne lit jamais le holdout. Les réglages sont dans la section `portfolio` du YAML.

## Représentation des arbres

```json
{"if": "D1.roc5_atr>p50",
 "then": {"if": "H4.chop14>p25", "then": {"leaf": "LONG"}, "else": {"leaf": "NO_TRADE"}},
 "else": {"leaf": "SHORT"}}
```

Grammaire des conditions :

| Exemple | Signification |
|---|---|
| `H4.rsi14>p75` | au-dessus du 75e percentile du train |
| `H4.rsi14>70` | seuil absolu |
| `D1.dist_ema200>0` | position relative à une autre série |
| `H1.macd_hist:rising` | pente sur 3 bougies |
| `H4.ema20_50:cross_up` | croisement de zéro dans les 5 dernières bougies |
| `H1.cal:session_london`, `H1.cal:day_friday` | calendrier |

Il n'existe pas de condition « en dessous » : c'est la branche `else` de la même condition.

## Export Pine Script v6

`uv run fxgen pine <tree_id>`, ou l'outil MCP `export_pine`, produit un indicateur qui colore le fond selon
le signal actif, marque les débuts de signaux, affiche la branche active sur la dernière bougie et fournit
des alertes. Les conditions H4/D1 passent par `request.security(..., lookahead = barmerge.lookahead_off)`,
qui reproduit l'alignement Python sur l'historique. Les valeurs peuvent différer légèrement si le flux de
données ou la frontière de session D1 de TradingView diffère de vos CSV.

## Serveur MCP

Serveur FastMCP (SDK officiel `mcp`, transport stdio). Il lit la base SQLite et lance les runs longs en
tâche de fond. La variable `FXGEN_CONFIG` choisit le YAML, et donc les données, la base et les sorties.

### Claude Code

```bash
claude mcp add fxgen \
  --env FXGEN_CONFIG=/chemin/absolu/fxgen/config/default.yaml \
  -- uv --directory /chemin/absolu/fxgen run fxgen-mcp
```

Vérifier avec `claude mcp list`. Ajouter `--scope user` pour l'avoir dans tous les projets.

### Claude Desktop

Dans `claude_desktop_config.json` (macOS : `~/Library/Application Support/Claude/`, Windows :
`%APPDATA%\Claude\`) :

```json
{
  "mcpServers": {
    "fxgen": {
      "command": "uv",
      "args": ["--directory", "/chemin/absolu/fxgen", "run", "fxgen-mcp"],
      "env": {"FXGEN_CONFIG": "/chemin/absolu/fxgen/config/default.yaml"}
    }
  }
}
```

Redémarrez Claude Desktop. Si `uv` n'est pas trouvé, mettez son chemin absolu (`which uv`).

### Outils

| Lecture | Action |
|---|---|
| `list_datasets()` | `run_evolution(config_overrides)` → `job_id` (les sections `splits` et `storage` sont verrouillées) |
| `list_conditions(family, limit, offset)` | `get_job_status(job_id)` / `cancel_job(job_id)` |
| `get_rankings(stage, top_n, sort_by)` | `backtest_tree(tree_json, pairs, timeframe, split)` (journalisé, compte dans le DSR) |
| `get_tree(tree_id)` | `run_robustness(tree_id)` (jamais sur le holdout) |
| `get_metrics(tree_id, split, pair)` | `export_pine(tree_id)` |
| `get_trades(tree_id, pair, limit, offset)` | `generate_report(tree_id)` |
| `get_robustness(tree_id)`, `compare_trees(tree_ids)`, `list_runs()` | `classify_strategy(tree_id)` |
| `get_portfolio(portfolio_id)` | `build_portfolio(run_ids, max_strategies, max_correlation, select_by, max_per_type, weighting)` |

L'étape holdout finale n'est volontairement pas exposée en MCP : elle se lance en CLI (`fxgen holdout`),
pour que ce soit une décision consciente.

Exemples de demandes à Claude :
- « Lance un run avec une profondeur 4 et 40 générations. »
- « Montre-moi le top 10 holdout trié par DSR. »
- « Compare les trois premiers arbres et dis-moi lesquels sont décorrélés. »
- « Teste cet arbre modifié sur la validation, uniquement sur EURUSD et GBPUSD. »
- « Construis un portefeuille de 4 stratégies avec une corrélation max de 0,4 et au plus une par type. »

## Architecture

```
src/fxgen/
  config.py            YAML -> pydantic (validation stricte, hash)
  data/                loader CSV, validation, alignement multi-TF, splits, resample, données synthétiques
  indicators/          primitives causales (sémantique Pine) + features par famille + calendrier
  conditions/          génération des conditions, seuils sur train, redondance
  market.py            features alignées par paire, matrices booléennes par split
  workspace.py         données + features + bibliothèque de conditions pour une config
  tree/                arbre, évaluation vectorisée, JSON, règles texte
  backtest/            moteur numba (sorties dépendantes du chemin), coûts, métriques
  ga/                  opérateurs, fitness robuste, partage de fitness, évolution parallèle
  validation/          walk-forward, permutation, DSR, sensibilité, Monte Carlo, holdout, batterie
  analysis/            type de stratégie, portefeuille décorrélé
  reporting/           classements, rapports HTML (arbre, portefeuille), export Pine
  storage/db.py        SQLite (runs, arbres, métriques, trades, robustesse, jobs, journal d'essais)
  pipeline.py          orchestration
  jobs.py              worker des runs en tâche de fond
  cli.py               commande `fxgen`
  mcp_server/server.py serveur MCP
```

## Limites connues

- Le backtest suppose une position par paire, des fills aux prix mid plus le spread, et un stop prioritaire
  quand stop et TP sont touchés dans la même bougie (hypothèse prudente).
- Les coûts par défaut sont des ordres de grandeur pour un compte raw spread de prop firm. Calibrez-les sur
  votre broker.
- La permutation coûte cher : 19 runs × (features + GA). Comptez environ 1 h 15 avec la config par
  défaut sur 4 cœurs (53 min mesurées avec l'ancienne bibliothèque de 49 indicateurs ; le calcul des
  indicateurs prend maintenant environ 85 s au lieu de 35 s par run).
- SuperTrend, Parabolic SAR, KAMA et Fisher sont récursifs : l'export Pine reprend les algorithmes de
  référence, mais leurs premières valeurs peuvent différer de TradingView tant que l'historique chargé
  n'est pas le même.
- Les données synthétiques servent à tester la plomberie. Les performances obtenues dessus ne disent rien
  sur les marchés réels.
