# fxgen — générateur génétique d'arbres de décision pour le swing trading FX

fxgen génère des stratégies FX sous forme de petits arbres de décision (profondeur 3–4), les fait évoluer
avec un algorithme génétique, puis **ne garde que celles qui survivent à une batterie de tests de robustesse
stricte**. La priorité est d'éviter le data-mining bias et le sur-apprentissage, pas d'afficher de beaux
backtests. Le pipeline est pilotable depuis Claude via un serveur MCP.

```
CSV OHLCV ─► validation ─► features (H1/H4/D1, ~150 indicateurs normalisés)
          ─► ~1000 conditions binaires (seuils = percentiles du TRAIN) ─► filtre de redondance
          ─► GA sur le TRAIN uniquement ─► candidats diversifiés
          ─► validation + walk-forward + permutation + DSR + sensibilité + Monte Carlo
          ─► finalistes ─► HOLDOUT (une seule fois) ─► classement, rapports HTML, Pine Script v6
```

## Installation

Python 3.11+ et [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run pytest            # ~80 tests, ~3 min
```

## Démarrage rapide sur données synthétiques

```bash
uv run fxgen synth                                   # 12 paires, H1/H4/D1, 2015-2024 -> data/synthetic/
uv run fxgen --config config/demo.yaml evolve        # GA + validation + robustesse (~5 min, 4 cœurs)
uv run fxgen --config config/demo.yaml holdout       # holdout des finalistes, une seule fois
```

Les sorties se trouvent dans `output/demo/<run_id>/` : `ranking.csv`, `ranking.json`, `reports/<tree>.html`
et `pine/<tree>.pine`.

Le jeu synthétique contient un **edge de momentum faible et volontaire** (`--edge 0.02`, à l'horizon de
5 jours), que le pipeline doit retrouver. Avec `uv run fxgen synth --edge 0`, les données sont du bruit pur
et rien ne doit survivre à la validation (`tests/test_validation.py::test_nothing_survives_on_pure_noise`).

`config/demo.yaml` réduit les budgets pour aller vite. Pour un vrai travail, utilisez `config/default.yaml`
(19 permutations, GA 200 × 30).

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
| `conditions` | percentiles, fenêtre de normalisation, seuil de redondance (\|corr\| > 0.9), taux d'activation min/max |
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
  - permutation.
- **Walk-forward du processus.** Le GA est relancé sur chaque fenêtre IS et son meilleur arbre est testé
  OOS. Le résultat (efficiency) est stocké avec le run.

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
| `get_robustness(tree_id)`, `compare_trees(tree_ids)`, `list_runs()` | |

L'étape holdout finale n'est volontairement pas exposée en MCP : elle se lance en CLI (`fxgen holdout`),
pour que ce soit une décision consciente.

Exemples de demandes à Claude :
- « Lance un run avec une profondeur 4 et 40 générations. »
- « Montre-moi le top 10 holdout trié par DSR. »
- « Compare les trois premiers arbres et dis-moi lesquels sont décorrélés. »
- « Teste cet arbre modifié sur la validation, uniquement sur EURUSD et GBPUSD. »

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
  reporting/           classements, rapport HTML, export Pine
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
- La permutation coûte cher : 19 runs × (features + GA). Comptez environ 1 h avec la config par défaut sur
  4 cœurs.
- Les données synthétiques servent à tester la plomberie. Les performances obtenues dessus ne disent rien
  sur les marchés réels.
