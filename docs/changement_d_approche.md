# Synthèse du projet IA & Application

## Jumeau numérique d'aéroport, DeepSDF et prédiction de dispersion de NOx

Ce document retrace le chemin suivi pendant le projet, les difficultés rencontrées, l'évolution de l'approche adoptée, ainsi que la version finale retenue.

Il ne s'agit pas ici de reprendre l'ensemble du rapport détaillé, mais de présenter de manière claire la logique du travail mené.

---

## 1. Point de départ du projet

Le sujet proposé portait sur la construction d'un **jumeau numérique géométrique d'aéroport** à l'aide d'une représentation implicite de type **DeepSDF**.

L'objectif de départ était de :

- récupérer des données liées à un aéroport ;
- reconstruire une scène 3D ;
- apprendre une représentation continue de cette géométrie ;
- réutiliser cette base géométrique dans un cadre plus applicatif, en lien avec le vent, les obstacles et la pollution.

Le projet s'inscrivait donc d'abord dans une logique de **reconstruction géométrique**, avec une finalité environnementale.

---

## 2. Première phase de travail

La première étape a consisté à mettre en place un pipeline correspondant à cette logique.

Les éléments suivants ont été développés :

- ingestion de données géospatiales ;
- extrusion 3D des bâtiments ;
- génération d'un dataset SDF ;
- entraînement d'un modèle DeepSDF ;
- visualisations ;
- génération d'un masque d'obstacles.

Cette partie repose notamment sur les fichiers :

- [config.yaml](../config.yaml)
- [data_ingestion.py](../src/data_ingestion.py)
- [sdf_sampling.py](../src/sdf_sampling.py)
- [deepsdf_model.py](../src/deepsdf_model.py)
- [train.py](../src/train.py)
- [visualize.py](../src/visualize.py)
- [generate_mask.py](../src/generate_mask.py)
- [deepsdf_airport_colab.ipynb](../notebooks/deepsdf_airport_colab.ipynb)

---

## 3. Limites de cette première approche

Malgré cette première implémentation, les résultats obtenus n'étaient pas assez solides pour constituer le cœur du rendu final.

Plusieurs étapes ont bien fonctionné :

- génération de maillages ;
- échantillonnage de points SDF ;
- entraînement du modèle ;
- production de sorties et de visualisations.

En revanche, le point essentiel était d'obtenir une géométrie suffisamment propre pour aller jusqu'à un masque d'obstacles exploitable.

Les principales limites observées ont été les suivantes :

- une reconstruction géométrique encore peu convaincante ;
- des rendus peu lisibles ;
- surtout, un masque final d'obstacles vide.

La chaîne existait donc bien, mais le résultat final n'était pas suffisamment fiable pour soutenir correctement la suite du projet.

---

## 4. Apport de la base fournie par le professeur

Dans un second temps, une base HDF5 a été fournie afin de simplifier le travail.

Cette base ne change pas le domaine du projet. Le travail reste dans le même cadre :

- aéroport ;
- géométrie ;
- vent ;
- émissions ;
- pollution ;
- concentration de NOx.

En revanche, elle remplace une grande partie de la phase la plus lourde en amont, en particulier :

- la construction du dataset à partir de données brutes ;
- la préparation de certains champs géométriques ;
- une partie importante de la mise en forme nécessaire avant l'entraînement final.

L'apport principal de cette base a donc été de permettre un recentrage du travail sur l'entraînement, l'évaluation et l'analyse d'un modèle de prédiction.

---

## 5. Évolution de l'approche

Le changement d'approche est venu d'un constat expérimental.

La progression du projet a été la suivante :

1. mise en place d'une première chaîne complète fondée sur DeepSDF ;
2. constat des limites de cette approche sur le résultat géométrique final ;
3. exploitation de la base fournie pour poursuivre le projet sous une forme plus robuste.

Cette évolution reste cohérente avec le sujet pour plusieurs raisons :

- la partie DeepSDF a bien été développée ;
- la base fournie s'inscrit dans le même cadre applicatif ;
- le travail reste centré sur l'environnement aéroportuaire et la dispersion ;
- la nouvelle direction permet d'aboutir à une partie finale réellement mesurable.

---

## 6. Travail réalisé à partir de la base fournie

À partir de cette base, un notebook dédié à la prédiction de la **dispersion de NOx** a été développé :

- [nox_dispersion_unet.ipynb](../notebooks/nox_dispersion_unet.ipynb)

Ce notebook prend en entrée des informations déjà structurées, notamment :

- le vent ;
- des champs SDF liés aux obstacles et à l'activité aéroportuaire ;
- les émissions.

Il vise ensuite à prédire :

- un champ 3D de concentration de NOx.

Le travail n'était donc plus centré sur l'apprentissage d'une géométrie implicite pure, comme dans la première phase, mais sur l'apprentissage d'une sortie physique à partir d'une base géométrique et météorologique déjà préparée.

---

## 7. Mise au point du notebook final

Le notebook n'a pas été opérationnel dès la première version. Plusieurs itérations ont été nécessaires avant d'obtenir un pipeline stable.

### 7.1 Difficultés techniques

Les premières difficultés étaient principalement des difficultés d'exécution :

- erreurs liées au format de précision utilisé sur le serveur ;
- cellules de visualisation qui interrompaient le notebook ;
- gestion des checkpoints à fiabiliser ;
- organisation des sorties à rendre plus propre.

Ces points ont été corrigés progressivement afin d'assurer une exécution stable du notebook.

### 7.2 Difficultés liées à l'entraînement

Une fois la partie technique stabilisée, la difficulté principale venait du comportement du modèle.

Les premiers entraînements montraient un apprentissage trop centré sur le fond :

- accuracy élevée ;
- macro-F1 faible ;
- apprentissage insuffisant des panaches.

Plusieurs modifications ont alors été apportées :

- pertes davantage centrées sur les zones utiles ;
- crops plus pertinents ;
- meilleure sélection du meilleur modèle ;
- validation plus représentative ;
- corrections sur certaines augmentations.

Cette phase a permis d'aboutir à une version du notebook exploitable sur le plan expérimental.

---

## 8. Version finale retenue

La version finale repose principalement sur :

- [nox_dispersion_unet.ipynb](../notebooks/nox_dispersion_unet.ipynb)
- [results/nox](../results/nox)

Un entraînement complet de 200 époques a été mené, avec sauvegarde des checkpoints, évaluation finale et génération des visualisations.

Le résultat obtenu n'est pas parfait, mais il est plus abouti que les premières versions.

### Points principaux

- le notebook s'exécute de manière stable ;
- l'entraînement reste cohérent sur toute sa durée ;
- l'évaluation finale est exploitable ;
- plusieurs scénarios non nuls donnent des résultats crédibles ;
- des figures d'analyse ont pu être produites sur la base du run final.

### Limites principales

- les cas `zero_case_proxy` restent les plus difficiles ;
- certaines classes intermédiaires de concentration sont encore mal séparées ;
- quelques outliers dégradent fortement certaines moyennes globales.

---

## 9. Éléments retenus pour le rendu

Les éléments principaux à conserver sont les suivants :

- [deepsdf_airport_colab.ipynb](../notebooks/deepsdf_airport_colab.ipynb)
- [nox_dispersion_unet.ipynb](../notebooks/nox_dispersion_unet.ipynb)
- [results/nox](../results/nox)
- [config.yaml](../config.yaml)
- [data_ingestion.py](../src/data_ingestion.py)
- [sdf_sampling.py](../src/sdf_sampling.py)
- [deepsdf_model.py](../src/deepsdf_model.py)
- [train.py](../src/train.py)
- [visualize.py](../src/visualize.py)
- [generate_mask.py](../src/generate_mask.py)
- [provided](../provided)

Cet ensemble permet de présenter :

- la première démarche mise en place ;
- les limites rencontrées ;
- le travail réalisé ensuite ;
- la version finale retenue.

---

## 10. Lecture d'ensemble du projet

Le projet peut être lu de la manière suivante :

Dans un premier temps, une chaîne DeepSDF complète a été mise en place afin de reconstruire une géométrie d'aéroport et d'aller jusqu'à un masque d'obstacles. Cette partie a bien été développée et testée, mais les résultats obtenus n'étaient pas assez solides pour constituer le cœur du rendu final. Dans un second temps, la base préparée fournie par l'enseignant a permis de recentrer le travail sur l'entraînement d'un modèle de prédiction de dispersion de NOx. C'est cette seconde partie qui a conduit au résultat final retenu.

Cette lecture permet de relier sans rupture :

- la première phase géométrique ;
- la base fournie par l'enseignant ;
- le modèle final de prédiction.

---

## 11. Conclusion

Le projet n'a pas suivi une trajectoire entièrement linéaire, mais il a gardé une logique claire du début à la fin.

La première phase, fondée sur DeepSDF, a permis de construire une base technique conforme à l'énoncé, sans toutefois fournir un résultat géométrique final suffisamment solide pour aller jusqu'au bout dans de bonnes conditions. À partir de la base fournie par le professeur, le travail a donc été réorienté vers une étape plus concrète et plus robuste : la prédiction de dispersion de NOx sur une base déjà préparée.

La version finale n'est pas parfaite, mais elle est :

- cohérente ;
- documentée ;
- techniquement défendable ;
- suffisamment aboutie pour constituer le rendu final du projet.
