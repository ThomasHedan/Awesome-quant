# fxgen

Générateur génétique d'arbres de décision pour le swing trading FX (H1/H4/D1). Il ne garde que les stratégies
qui survivent à une batterie stricte de tests anti sur-apprentissage, et il est pilotable depuis Claude (MCP).

```
CSV ─► 180 indicateurs × H1/H4/D1 ─► ~7600 conditions (seuils = percentiles du train) ─► filtre de redondance
    ─► GA sur le train ─► validation, walk-forward, permutation, DSR, sensibilité, Monte Carlo, prop firm
    ─► finalistes ─► holdout (une fois) ─► classement, rapports HTML, Pine v6, type, portefeuille
```

## Démarrage

Python 3.11+ et [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev && uv run pytest                 # ~100 tests, ~7 min
uv run fxgen synth                                   # données synthétiques -> data/synthetic/
uv run fxgen --config config/demo.yaml evolve        # ~10 min sur 4 cœurs
uv run fxgen --config config/demo.yaml holdout
uv run fxgen --config config/demo.yaml portfolio
```

## Commandes

Toutes acceptent `--config FICHIER` et `--set section.cle=valeur`.

| Commande | Rôle |
|---|---|
| `synth [--edge 0.02]` | données synthétiques ; `--edge 0` = bruit pur (rien ne doit survivre) |
| `lse-download [--estimate]` | télécharge le forex de London Strategic Edge (voir Données) |
| `datasets` | qualité des CSV : plages, trous, doublons, bougies réparées |
| `evolve` | GA + validation + robustesse + finalistes (sans holdout) |
| `holdout [RUN_ID]` | holdout des finalistes, une seule fois par run, puis rapports et Pine |
| `report` / `pine` / `classify TREE_ID` | rapport HTML / export Pine v6 / type de stratégie |
| `portfolio [--runs ...]` | portefeuille de stratégies non corrélées |
| `challenge --tree ID` / `--portfolio ID` | simulateur de challenge prop firm |
| `runs` / `indicators` | liste des runs / liste des indicateurs en Markdown |

Sorties dans `output/<run_id>/` : `ranking.csv|json`, `reports/*.html`, `pine/*.pine`.
Portefeuilles dans `output/portfolios/`. Tout est aussi stocké en SQLite (`storage.database`).

## Données

| Source | Config | Notes |
|---|---|---|
| Synthétique | `config/demo.yaml` | 12 paires 2015-2024, edge de momentum 5 jours implanté. Sert à tester la plomberie, pas le marché. |
| [London Strategic Edge](https://londonstrategicedge.com/data) | `config/lse.yaml` | gratuit, clé requise : `export LSE_API_KEY=...` puis `fxgen lse-download` |
| Tes CSV | section `data` | un fichier par paire et timeframe (`{pair}_{timeframe}.csv`) |

**LSE** :
- seul le H1 est téléchargé ; H4 et D1 sont reconstruits localement (bascule journalière à 22 h UTC) ;
- plan gratuit : environ 3 paires × 10 ans par jour, donc environ 4 jours pour 12 paires ;
- relancer la commande reprend où elle s'était arrêtée (tranches vérifiées par SHA-256) ;
- `config/lse.yaml` : train 2018-2022, validation 2023-2024, holdout depuis 2025 ;
- clé personnelle, à ne jamais committer ni partager (conditions du site).

**Tes CSV** :

```yaml
data:
  directory: /chemin/csv
  columns: {time: Date, open: Open, high: High, low: Low, close: Close, volume: Volume}  # volume: null si absent
  source_timezone: Europe/Paris   # converti en UTC
  timestamp_label: open           # l'horodatage = ouverture ou clôture de la bougie
```

## Configuration

`config/default.yaml` est commenté. Sections :

| Section | Contenu |
|---|---|
| `splits` | train < validation < holdout < end |
| `conditions` | percentiles, redondance (\|corr\| > 0.9), filtres : `families`, `exclude_features`, `include_conditions`, `exclude_conditions` (globs, ex. `"D1.*"`, `"*:cross_*@*"`) |
| `tree`, `ga` | profondeur max, population, générations, tournoi, crossover, mutation, élitisme, niching, `n_jobs` |
| `backtest`, `costs` | stop/TP en ATR (`atr_timeframe: D1`), sortie temporelle ou au changement de branche, risque par trade ; spread par paire, commission, swap |
| `fitness` | trades min/max, pénalité de complexité, dépendance à une paire ou à une période |
| `validation` | walk-forward, permutation, sensibilité, Monte Carlo, `dsr_min`, candidats, finalistes |
| `prop_firm` | DD max, perte journalière max, risque minimal exigé ; `challenge` : phases (objectif, durée, jours tradés min), DD fixe ou suiveur, frais, grille de risque, décote de l'edge |
| `portfolio` | nombre de stratégies, corrélation max, critère, quota par type, pondération |

## Garanties anti-biais

| Garantie | Mécanisme |
|---|---|
| Pas de lookahead | décision à la clôture de `t`, exécution à l'ouverture de `t+1` ; H4/D1 utilisables seulement une fois clôturées. Tests : corrompre le futur ne change pas le passé. |
| Seuils hors échantillon | percentiles calculés sur le train, toutes paires confondues, sur des indicateurs normalisés |
| Holdout verrouillé | le GA le refuse ; lu une seule fois par run ; tout accès manuel est journalisé et compte dans le DSR |
| Nombre d'essais | DSR (Bailey & López de Prado) avec N = tous les arbres évalués + backtests manuels + accès holdout |
| Bruit | permutation : tout le pipeline est relancé sur des bougies mélangées (19 runs, p ≤ 0,05) |

**Batterie par arbre.** Un arbre survit s'il passe tous les tests :
- validation positive ;
- DSR train ≥ `dsr_min` ;
- walk-forward majoritairement positif ;
- sensibilité : percentiles ±5 et sorties ×0,8 / ×1,2 gardent au moins 50 % du Sharpe ;
- Monte Carlo (DD à 95 %) et perte journalière compatibles avec la prop firm ;
- permutation.

## Arbres et conditions

```json
{"if": "D1.roc5_atr>p50", "then": {"leaf": "LONG"}, "else": {"if": "H4.rsi14:cross_down@70", "then": {"leaf": "SHORT"}, "else": {"leaf": "NO_TRADE"}}}
```

| Syntaxe | Sens |
|---|---|
| `H4.rsi14>p75`, `H4.rsi14>70`, `D1.dist_ema200>0` | au-dessus d'un percentile du train / d'un niveau / de 0 |
| `H1.macd_hist:rising` | en hausse sur 3 bougies |
| `H4.ema20_50:cross_up`, `H4.rsi14:cross_down@70` | a croisé 0 / un niveau dans les 5 dernières bougies |
| `H1.inside_bar>0.5`, `H1.inside_bar:cross_up@0.5` | pattern présent / apparu dans les 5 dernières bougies |
| `H1.cal:session_london`, `H1.cal:month_end` | calendrier (21 règles) |

« En dessous » n'existe pas : c'est la branche `else`. La liste des indicateurs est dans
[`docs/INDICATORS.md`](docs/INDICATORS.md).

## Type de stratégie et portefeuille

- **Type** (`classify`, `classify_strategy`) : trend following, momentum, breakout, mean reversion ou mixte.
  Le classement combine les règles (indicateurs utilisés, dans le sens du mouvement ou contre lui) et le
  comportement des trades (entrées dans ou contre le mouvement de la veille et des 5 derniers jours).
- **Portefeuille** (`portfolio`, `build_portfolio`) :
  - part des survivants triés par DSR de validation ;
  - ajoute une stratégie seulement si sa corrélation journalière avec chaque membre reste sous le plafond, et
    si son type n'a pas atteint son quota ;
  - donne les pondérations, les métriques et le risque par trade compatible avec la prop firm ;
  - ne lit jamais le holdout.

## Simulateur de challenge prop firm

`fxgen challenge --tree ID` (ou `--portfolio ID`, ou l'outil MCP `simulate_challenge`) rejoue des milliers de
challenges à partir des journées du train et de la validation, tirées par blocs de 5 jours.

**Règles** : objectifs des phases (8 % puis 5 % par défaut), perte journalière max, DD max (fixe ou
suiveur), jours tradés minimum, durée limite, puis période de compte financé.

**Résultat, pour chaque risque par trade** :
- probabilité de réussir chaque phase et le challenge complet ;
- cause des échecs (perte journalière, DD, temps) ;
- jours médians pour réussir ;
- coût attendu en frais ;
- probabilité de garder le compte financé.

Le meilleur risque est indiqué.

**Prudence** :
- `edge_haircut: 0.5` retire la moitié du rendement moyen du backtest, toujours optimiste ;
- les pertes latentes intrajournalières ne sont pas modélisées, donc le vrai risque journalier est un peu plus
  élevé.

## Export Pine v6

L'export est un indicateur TradingView :
- fond coloré selon le signal ;
- marqueurs d'entrée, branche active, alertes ;
- H4/D1 via `request.security(..., lookahead_off)`.

Petits écarts possibles avec le Python :
- si le flux de données ou la session D1 de TradingView diffère de tes CSV ;
- sur les premières bougies de SuperTrend, SAR, KAMA et Fisher.

## Serveur MCP

```bash
claude mcp add fxgen --env FXGEN_CONFIG=/chemin/fxgen/config/default.yaml -- uv --directory /chemin/fxgen run fxgen-mcp
```

Claude Desktop (`claude_desktop_config.json`) :

```json
{"mcpServers": {"fxgen": {"command": "uv", "args": ["--directory", "/chemin/fxgen", "run", "fxgen-mcp"],
                          "env": {"FXGEN_CONFIG": "/chemin/fxgen/config/default.yaml"}}}}
```

| Lecture | Action |
|---|---|
| `list_datasets`, `list_conditions`, `list_runs` | `run_evolution` (tâche de fond) → `get_job_status`, `cancel_job` |
| `get_rankings`, `get_tree`, `get_metrics`, `get_trades` | `backtest_tree` (journalisé ; compte dans le DSR) |
| `get_robustness`, `compare_trees` | `run_robustness`, `classify_strategy` |
| `get_portfolio` | `build_portfolio`, `simulate_challenge`, `export_pine`, `generate_report` |

Restrictions côté MCP :
- les sections `splits` et `storage` ne peuvent pas être modifiées ;
- l'étape holdout finale ne se lance qu'en CLI.

## Architecture

| Module | Rôle |
|---|---|
| `data/` | chargement et validation CSV, alignement multi-timeframe, splits, resample, synthétique, téléchargeur LSE |
| `indicators/`, `conditions/` | indicateurs causaux (sémantique Pine), conditions, seuils sur train, redondance |
| `market.py`, `workspace.py` | indicateurs alignés par paire, vues par split, bibliothèque de conditions |
| `tree/`, `backtest/` | arbres (JSON, règles) ; moteur numba, coûts, métriques |
| `ga/`, `validation/` | GA, fitness, niching ; robustesse, holdout |
| `analysis/`, `reporting/` | type de stratégie, portefeuille ; classements, HTML, Pine |
| `storage/`, `pipeline.py`, `jobs.py`, `cli.py`, `mcp_server/` | SQLite, orchestration, tâches de fond, CLI, MCP |

## Performances et limites

- **Run complet** (`default.yaml`, 12 paires × 10 ans en H1) : environ 1 h 15 estimée sur 4 cœurs. Mémoire :
  environ 3 Go en régime établi, 4,5 Go au pic. Les filtres `conditions` réduisent les deux.
- **Backtest** : une position par paire. Si stop et TP sont touchés dans la même bougie, le stop est
  appliqué en premier (hypothèse prudente).
- **Coûts** : ceux par défaut sont des ordres de grandeur ; à calibrer sur ton broker.
