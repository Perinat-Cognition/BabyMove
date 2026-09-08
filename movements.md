# Codage des mouvements du nourrisson

## 1. Principe général

Les mouvements sont regroupés en 3 catégories :

* **Step** : mouvements avant/arrière des bras et des jambes.
* **Kick** : mouvements haut/bas des jambes.
* **Pump** : mouvements haut/bas des bras.

Chaque mouvement est défini par :

```text
Start = début du mouvement
Stop  = fin du mouvement
```

Les mouvements sont donc codés sous 6 formes :

```text
Step  → Front / Back
Kick  → Up / Down
Pump  → Up / Down
```

---

## 2. Step

Le Step correspond à un déplacement horizontal du membre par rapport au centre de masse.

### Jambes

**Front** = le genou se rapproche du centre de masse.

**Back** = le genou s'éloigne du centre de masse.

| Mouvement  | Start                          | Stop                         |
| ---------- | ------------------------------ | ---------------------------- |
| Step Front | début de la flexion de hanche  | fin de la flexion de hanche  |
| Step Back  | début de l'extension de hanche | fin de l'extension de hanche |

Une flexion/extension du genou ou de la cheville peut faire partie du même mouvement.

### Bras

**Front** = la main s'éloigne du centre de masse vers l'avant.

**Back** = la main revient vers le centre de masse.

Le mouvement peut être produit par l'épaule et/ou le coude.

---

## 3. Kick

Le Kick concerne uniquement les jambes.

**Up** = la cheville monte principalement par flexion du genou, sans mouvement important de la hanche.

**Down** = la cheville redescend principalement par extension du genou.

| Mouvement | Start                         | Stop                        |
| --------- | ----------------------------- | --------------------------- |
| Kick Up   | début de la flexion du genou  | fin de la flexion du genou  |
| Kick Down | début de l'extension du genou | fin de l'extension du genou |

Le Kick Up ressemble généralement à un mouvement « talon-fesse ».

---

## 4. Pump

Le Pump concerne uniquement les bras.

**Up** = la main ou le coude monte.

**Down** = la main ou le coude descend.

Le mouvement peut être produit par l'épaule (abduction, adduction, rotation) et éventuellement par le coude.

| Mouvement | Start                           | Stop                          |
| --------- | ------------------------------- | ----------------------------- |
| Pump Up   | début du mouvement vers le haut | fin du mouvement vers le haut |
| Pump Down | début du mouvement vers le bas  | fin du mouvement vers le bas  |

---

# 5. Règles de regroupement

L'objectif est de ne pas découper artificiellement un même mouvement dynamique en plusieurs mouvements.

Ces règles s'appliquent à **Front/Back**, **Up/Down**.

### Règle 1 — Mouvement faible suivi rapidement d'un mouvement significatif

Si :

```text
mouvement faible → arrêt < 15 frames → mouvement significatif
```

alors les deux sont regroupés.

```text
Start = Start du premier
Stop  = Stop du second
```

Le premier mouvement est donc conservé même s'il était trop faible pour être codé seul.

---

### Règle 2 — Deux mouvements dans le même sens

Si deux mouvements vont dans le même sens et sont séparés par moins de **60 frames** :

```text
Mouvement A → arrêt < 60 frames → Mouvement B
```

ils sont regroupés en un seul :

```text
Start = Start A
Stop  = Stop B
```

---

### Règle 3 — Petit mouvement opposé au milieu

Si :

```text
Mouvement A
    ↓
petit mouvement opposé
    ↓
Mouvement A
```

et que les deux temps d'arrêt sont < **15 frames**, le petit mouvement opposé est ignoré.

Exemple :

```text
Front → petit Back → Front
```

devient :

```text
Front
Start = Start du premier Front
Stop  = Stop du second Front
```

Le petit mouvement opposé doit avoir une amplitude **très inférieure** aux deux mouvements qui l'entourent.

---

# 6. Règles spécifiques au Step

Pour les mouvements **Front** :

### Début anticipé

Si une flexion du genou/cheville précède l'avancée du genou et fait partie du même élan :

```text
flexion genou → avancée genou
```

le `Start Front` est placé au début de la flexion du genou/cheville.

Pour le bras :

```text
avancée coude → avancée main
```

le `Start Front` est placé au début de l'avancée du coude.

### Fin retardée

Si une flexion du genou/cheville suit l'avancée du genou et appartient au même élan :

```text
avancée genou → flexion genou
```

le `Stop Front` est placé à la fin de la flexion.

Les mêmes principes sont appliqués symétriquement au **Back** avec les extensions.