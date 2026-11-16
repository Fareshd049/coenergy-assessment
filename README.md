# coEnergy — prédiction multi-indicateurs de la performance thermique

Modèle PyTorch qui, à partir de 8 h de mesures à la minute de la température intérieure $T_{int}(t)$ et de la sollicitation thermique $U(t)$, prédit six indicateurs d'un bâtiment :

`[HLC, HLC_ext, HLC_mit, tau_b, tau_n, tau_inf]`

Il fournit aussi un intervalle de confiance calibré pour chacun. Les relations physiques ($\mathrm{HLC} = \mathrm{HLC}_{ext} + \mathrm{HLC}_{mit}$, positivité, $\tau_n \le \tau_\infty \le \tau_b$) sont garanties par construction.

**Le rapport complet est le notebook [`notebooks/report.ipynb`](notebooks/report.ipynb)** (en français) : EDA, choix de modélisation, résultats et analyse critique.

## Résultats sur le jeu de test

Évaluation unique sur 30 000 simulations jamais vues pendant l'entraînement ni la sélection du modèle.

| Indicateur | $R^2$ | Erreur relative moyenne | P95 de l'erreur absolue | Couverture de l'intervalle à 95 % |
|---|---|---|---|---|
| HLC | 0,39 | 8,5 % | 48 W/K | 95,7 % |
| HLC_ext | 0,32 | 10,3 % | 27 W/K | 96,2 % |
| HLC_mit | 0,26 | 15,9 % | 44 W/K | 95,7 % |
| tau_b | 0,08 | 15,3 % | 19 h | 95,4 % |
| tau_n | 0,22 | 12,5 % | 3,4 h | 95,6 % |
| tau_inf | 0,06 | 18,3 % | 14 h | 95,3 % |

Les performances plafonnent : cinq leviers indépendants (architectures, ×5 données, mesures parfaites, ensemble, modèles spécialistes) ne déplacent pas le $R^2$ du HLC de plus de 0,02. La limite vient de l'information contenue dans un essai PRBS de 8 h dont l'état initial des parois est inconnu (sections 6.6 et 8 du rapport).

## Installation

Le projet utilise [uv](https://docs.astral.sh/uv/). Les versions sont figées dans `uv.lock`.

```bash
uv sync
uv run pytest            # tests unitaires
uv run ruff check .      # style
```

**Données.** Placer les 20 fichiers `batch_1.pt` … `batch_20.pt` fournis par coEnergy dans `data/quentin_house/`. Ce dossier n'est pas versionné.

## Lire le rapport

```bash
uv run --with jupyter jupyter lab notebooks/report.ipynb
```

Le notebook **ne ré-entraîne rien**. Les sections 1 à 4 chargent un échantillon des données ; les sections 5 à 8 relisent les résultats enregistrés dans `artifacts/` :
- les métriques de chaque entraînement : `artifacts/runs/<nom>/val_metrics.json` ;
- les tableaux et figures des évaluations : `artifacts/eval*/`.

Il s'exécute en quelques minutes sur CPU.

## Reproduire les expériences

Chaque expérience est entièrement décrite par un fichier YAML de `configs/`.

```bash
# Entraînement (GPU recommandé ; le modèle final a été entraîné sur GPU Kaggle T4)
uv run python scripts/train.py --config configs/final_700k_noise005.yaml --data-folder data/quentin_house

# Ablations et variantes d'architecture (section 6.1)
uv run python scripts/train.py --config configs/ablations/abl_tcn.yaml --data-folder data/quentin_house
uv run python scripts/rocket_baseline.py --config configs/full.yaml --data-folder data/quentin_house

# Évaluation complète sur la validation, puis une seule fois sur le test
uv run python scripts/evaluate.py --runs artifacts/runs/cnn_gru_share_nll --split val --data-folder data/quentin_house
uv run python scripts/evaluate.py --runs artifacts/runs/final_700k_noise005 --split test \
    --data-folder data/quentin_house --no-unseen --noise-levels 0 0.02 0.05 0.1

# Vérifications du plafond (section 6.6)
uv run python scripts/ensemble_eval.py --data-folder data/quentin_house
uv run python scripts/specialist_eval.py --data-folder data/quentin_house
```

`notebooks/kaggle_train.ipynb` enchaîne ces entraînements sur Kaggle. Les listes d'expériences sont en tête du notebook ; il clone le dépôt avec un jeton stocké dans les *secrets* Kaggle.

**Notes**
- Les checkpoints enregistrent le chemin des données utilisé à l'entraînement : passer `--data-folder` pour évaluer sur une autre machine.
- Environnements utilisés : torch 2.11 sur Kaggle (entraînement), torch de `uv.lock` en local (évaluation, rapport). Les résultats ne dépendent pas de cette différence de version au-delà du bruit numérique.
- Graines fixées (`seed` dans chaque YAML). Sur GPU, la précision mixte et cuDNN ne garantissent pas une reproductibilité bit à bit.

## Organisation du code

```
src/hlcnet/            package Python, typé et documenté
  targets.py           ordre canonique des 6 indicateurs, familles, unités
  data.py              chargement des fichiers, découpage par fichier, normalisation, bruit capteur
  eda.py               statistiques et figures de l'analyse exploratoire
  identifiability.py   expériences naturelles, régression ridge, ACP (section 3.6)
  models.py            CNN-GRU, TCN, FiLM-GRU, MLP ; tête physique ; tête d'incertitude
  losses.py            beta-NLL gaussienne, MSE, Huber ; pondération par famille
  train.py             boucle d'entraînement, arrêt précoce, checkpoints
  metrics.py           métriques par indicateur et par famille, contraintes, couverture
  analysis.py          calibration, biais log-normal, bruit, sous-groupes, cartes d'excitation, ensembles
  specialist.py        modèles spécialistes par excitation (section 6.6 c)
  rocket.py            référence ROCKET (noyaux aléatoires)
  report.py            lecture des résultats enregistrés pour le notebook
  config.py, utils.py  configuration YAML, graines, périphérique
scripts/               points d'entrée en ligne de commande (train, evaluate, ...)
configs/               une expérience = un fichier YAML
tests/                 tests unitaires (pytest)
notebooks/             report.ipynb (rapport), kaggle_train.ipynb (entraînement GPU)
artifacts/             métriques, figures et checkpoints des expériences
```
