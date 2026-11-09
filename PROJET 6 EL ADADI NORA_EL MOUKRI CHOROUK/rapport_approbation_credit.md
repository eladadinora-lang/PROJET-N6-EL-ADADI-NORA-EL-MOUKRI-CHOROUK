![Université Hassan 1er — ENCG Settat](figures/banniere_uh1_encg.jpg)

# Apprentissage automatique appliqué à l'approbation de crédit

**Projet 6 — Rapport d'analyse**

**ENCG Settat — Filière Finance, S7 G2**

Réalisé par : **NORA EL ADADI** et **CHOROUK EL MOUKRI**

Encadré par : **M. LAGHLIMI**

Année universitaire 2026-2027

---

## Sommaire

1. Introduction et problématique
2. Exploration des données
3. Préparation et modélisation
4. Évaluation du modèle retenu
5. Interprétation du modèle
6. Limites de l'étude
7. Conclusion

---

## 1. Introduction et problématique

Quand une banque reçoit une demande de crédit, elle doit décider : accorder ou refuser. Si elle accorde trop facilement, certains clients ne rembourseront pas. Si elle refuse trop souvent, elle perd de bons clients. Aujourd'hui, les banques utilisent de plus en plus des modèles informatiques pour les aider à décider, et c'est ce que nous avons voulu tester dans ce projet.

Nous avons utilisé le jeu de données Credit Approval (UCI). Il contient 690 demandes de carte de crédit, avec 15 informations sur chaque demande (A1 à A15) et la décision finale (A16 : « + » si le crédit est accordé, « − » s'il est refusé). Les noms des variables ont été cachés pour protéger les clients. On sait seulement que 6 variables sont des nombres et 9 sont du texte.

Pour chaque demande, on connaît déjà la bonne réponse (accordé ou refusé), et le modèle apprend à partir de ces exemples. C'est ce qu'on appelle l'apprentissage supervisé. Comme la réponse n'a que deux valeurs possibles, on parle de classification binaire.

La question à laquelle nous essayons de répondre est la suivante : *un modèle peut-il prendre les mêmes décisions que la banque, et peut-il l'aider à éviter l'erreur la plus grave, c'est-à-dire accorder un crédit à quelqu'un qui aurait dû être refusé ?*

## 2. Exploration des données

### 2.1 Qualité des données

Les données sont presque complètes : seules 37 demandes sur 690 (environ 5 %) ont une information manquante. Par contre, les variables n'ont pas du tout la même taille : A2 va de 14 à 80, alors que A15 va de 0 à 100 000. Pour A15, la moyenne est de 1 017 mais la médiane n'est que de 5, ce qui veut dire que quelques clients ont des valeurs très grandes. Avant d'entraîner les modèles, il faudra donc ramener toutes les variables à la même échelle.

### 2.2 Répartition des décisions

![Figure 1 – Nombre de demandes par décision](figures/cell16.png)

*Figure 1 – Répartition des demandes selon la décision*

Il y a 383 refus (55,5 %) et 307 accords (44,5 %). Les deux groupes ont à peu près la même taille. Ce chiffre nous donne un point de départ : un modèle qui refuserait tout le monde aurait déjà 55,5 % de bonnes réponses. Nos modèles doivent donc faire mieux que ça.

### 2.3 Variables numériques

![Figure 2 – Distribution des variables numériques selon la décision](figures/cell20.png)

*Figure 2 – Distribution des variables numériques selon la décision (échelle log pour les variables asymétriques)*

| Médiane | A2 | A3 | A8 | A11 | A14 | A15 |
|---|---|---|---|---|---|---|
| Accordé | 30,50 | 4,46 | 2,00 | 3,0 | 120,0 | 221,0 |
| Refusé | 27,33 | 2,21 | 0,42 | 0,0 | 167,5 | 1,0 |

Plus les deux courbes sont éloignées, plus la variable aide à faire la différence entre accordés et refusés. A8, A11 et A15 sont nettement plus élevées chez les clients acceptés. Par exemple, la médiane d'A11 est de 0 chez les refusés et de 3 chez les acceptés. A3 va dans le même sens, mais l'écart est plus petit. Pour A2 et A14, les deux groupes se ressemblent beaucoup, donc ces variables aident peu. Enfin, les variables sont peu liées entre elles (le lien le plus fort est de 0,40, entre A2 et A8) : chacune apporte une information différente.

### 2.4 Variables texte

![Figure 3 – Taux d'accord par modalité](figures/cell24.png)

*Figure 3 – Taux d'accord pour chaque valeur des variables texte (pointillés : taux moyen de 44,5 %)*

Ce graphique montre le pourcentage de crédits accordés pour chaque valeur. A9 est clairement la variable la plus importante : quand A9 = t, 79 % des demandes sont accordées, et quand A9 = f, seulement 7 %. A10 compte aussi (71 % contre 25 %). Les autres variables changent peu le taux d'accord.

## 3. Préparation et modélisation

### 3.1 Préparation des données

Nous avons séparé les données en deux parties : 75 % pour entraîner les modèles (517 demandes) et 25 % pour les tester à la fin (173 demandes). Les deux parties ont la même proportion d'accords (44,5 %). La partie test n'a servi qu'une seule fois, tout à la fin.

Ensuite, nous avons préparé les données : les informations manquantes ont été remplacées par la valeur la plus courante, les variables texte ont été transformées en colonnes de 0 et de 1, et les nombres ont été mis à la même échelle.

### 3.2 Comparaison des modèles

Nous avons comparé cinq modèles. Pour chacun, les données d'entraînement ont été coupées en 5 parties, et chaque partie a servi une fois à tester le modèle (validation croisée). Nous avons regardé deux indicateurs : l'exactitude, qui est le pourcentage de bonnes réponses, et l'AUC, une note entre 0,5 (le modèle répond au hasard) et 1 (le modèle ne se trompe jamais) qui mesure si le modèle sait reconnaître un bon dossier d'un mauvais.

| Modèle | Exactitude moyenne | Écart-type | AUC moyenne |
|---|---|---|---|
| Forêt aléatoire | 0,876 | 0,027 | 0,925 |
| Régression logistique | 0,849 | 0,030 | 0,916 |
| K plus proches voisins (k = 5) | 0,830 | 0,014 | 0,893 |
| Arbre de décision (profondeur 4) | 0,816 | 0,020 | 0,879 |
| Référence (refus systématique) | 0,555 | 0,002 | 0,500 |

![Figure 4 – Comparaison des modèles](figures/cell41.png)

*Figure 4 – Exactitude moyenne de chaque modèle*

La référence refuse tout le monde et obtient 55,5 %. Tous les vrais modèles font beaucoup mieux, ce qui montre que les données contiennent de l'information utile. La forêt aléatoire est la meilleure, avec environ 88 % de bonnes réponses, devant la régression logistique (environ 85 %). L'écart entre les deux reste petit.

## 4. Évaluation du modèle retenu

### 4.1 Résultats sur les données de test

Nous avons entraîné la forêt aléatoire sur toutes les données d'entraînement, puis nous l'avons testée une seule fois sur les 173 demandes mises de côté. Elle obtient 90,2 % de bonnes réponses et une AUC de 0,949.

| Classe | Précision | Rappel | F1-score | Effectif |
|---|---|---|---|---|
| Refusé | 0,90 | 0,93 | 0,91 | 96 |
| Accordé | 0,91 | 0,87 | 0,89 | 77 |

### 4.2 Matrice de confusion

![Figure 5 – Matrice de confusion](figures/cell48.png)

*Figure 5 – Matrice de confusion sur les données de test*

Ce tableau montre les bonnes réponses et les erreurs du modèle. Sur 173 demandes, il se trompe 17 fois :

- 7 fois, il accorde un crédit qui aurait dû être refusé (faux positif). C'est l'erreur la plus grave, car la banque prend un risque de ne pas être remboursée ;
- 10 fois, il refuse un bon client (faux négatif). Là, la banque perd un client.

Le modèle retrouve 93 % des refus et 87 % des accords. Il est donc plutôt prudent, ce qui est une bonne chose pour une banque.

### 4.3 Courbe ROC et choix du seuil

![Figure 6 – Courbe ROC](figures/cell52.png)

*Figure 6 – Courbe ROC de la forêt aléatoire*

| Seuil | Exactitude | Faux positifs (risque) | Faux négatifs (clients perdus) |
|---|---|---|---|
| 0,3 | 0,890 | 17 | 2 |
| 0,4 | 0,896 | 12 | 6 |
| 0,5 | 0,902 | 7 | 10 |
| 0,6 | 0,867 | 6 | 17 |
| 0,7 | 0,838 | 4 | 24 |

Plus la courbe est proche du coin en haut à gauche, meilleur est le modèle. Ici, elle en est très proche (AUC d'environ 0,95), donc le modèle est bon.

Le modèle donne pour chaque demande une probabilité d'accord. Par défaut, il accorde le crédit si cette probabilité dépasse 0,5 : c'est le seuil. Le tableau montre que si on monte le seuil, le modèle devient plus prudent : il y a moins de crédits risqués accordés, mais plus de bons clients refusés. Par exemple, avec un seuil de 0,7, on passe de 7 à 4 crédits risqués, mais de 10 à 24 clients perdus. Le choix du seuil dépend donc de ce qui coûte le plus cher à la banque.

### 4.4 Courbe d'apprentissage

![Figure 7 – Courbe d'apprentissage](figures/cell56.png)

*Figure 7 – Courbe d'apprentissage de la forêt aléatoire*

Cette courbe montre si le modèle s'améliore quand on lui donne plus de données. Le score d'entraînement est toujours de 1 : la forêt apprend par cœur les demandes qu'elle a déjà vues. Le score de validation monte doucement, de 0,85 à 0,88. L'écart entre les deux courbes montre que le modèle apprend un peu trop par cœur. Avec plus de données, il pourrait encore s'améliorer un peu.

## 5. Interprétation du modèle

### 5.1 Importance des variables

![Figure 8 – Importance des variables](figures/cell60.png)

*Figure 8 – Importance des variables dans la forêt aléatoire*

Ce graphique montre les variables que le modèle utilise le plus pour décider. A9 est de loin la plus importante (environ 29 %). Ensuite viennent A10 (environ 10 %), puis A8, A6, A15, A3, A11 et A2 (entre 6 et 9 %). A1, A4, A5, A12 et A13 ne servent presque à rien.

Nous avons vérifié le poids de A9 avec une demande inventée. Le modèle lui donne une probabilité d'accord de 0,97. Si on change seulement A9 de « t » à « f », la probabilité tombe à 0,20.

### 5.2 Une règle simple fait-elle aussi bien ?

Puisque A9 est si importante, nous avons testé une règle très simple : accorder si A9 = t, refuser sinon.

| | Exactitude (test) | Faux positifs | Faux négatifs |
|---|---|---|---|
| Règle « A9 = t » | 0,908 | 15 | 1 |
| Forêt aléatoire | 0,902 | 7 | 10 |

La règle simple obtient presque le même pourcentage de bonnes réponses que la forêt. Mais elle accorde 15 crédits risqués, contre 7 pour la forêt, donc deux fois plus. La forêt est meilleure sur l'erreur la plus grave pour la banque. En plus, elle donne une probabilité, ce qui permet de régler le seuil.

## 6. Limites de l'étude

Nos résultats ont plusieurs limites :

- les variables sont anonymes : on ne sait pas ce que veulent dire A9 ou A10, donc on ne peut pas expliquer les décisions du modèle à un client ;
- le modèle apprend les décisions passées de la banque, pas les remboursements réels. Si la banque s'est trompée dans le passé, le modèle reproduit ses erreurs ;
- il y a peu de données : avec 173 demandes de test, une seule erreur change le résultat d'environ 0,6 point. La petite différence entre la forêt (90,2 %) et la règle A9 (90,8 %) ne veut donc pas dire grand-chose ;
- les données sont anciennes et ne correspondent pas à un pays ou un marché précis ;
- nous n'avons pas cherché les meilleurs réglages pour chaque modèle, et la forêt apprend un peu trop par cœur ;
- nous ne connaissons pas le coût réel d'un crédit non remboursé ni d'un client perdu, donc on ne peut pas choisir le meilleur seuil.

## 7. Conclusion

Ce projet montre qu'un modèle peut reproduire assez bien les décisions de crédit d'une banque. La forêt aléatoire est le meilleur modèle : elle donne environ 90 % de bonnes réponses sur des demandes qu'elle n'avait jamais vues.

Pour répondre à notre question, le pourcentage de bonnes réponses ne suffit pas. Une règle simple basée seulement sur A9 fait presque aussi bien, mais elle accorde deux fois plus de crédits risqués. L'intérêt de la forêt est qu'elle limite ces erreurs graves et qu'elle donne une probabilité, que la banque peut utiliser pour être plus ou moins prudente.

Ce travail reste un premier essai, car les variables sont anonymes, il y a peu de données et le modèle apprend les anciennes décisions de la banque. Pour l'utiliser vraiment, il faudrait des données récentes, connaître le coût de chaque erreur et mieux régler les modèles. Même dans ce cas, ce type de modèle doit rester une aide pour le conseiller crédit, pas le remplacer.
