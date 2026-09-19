# Stratégies de découpage pour les articles scientifiques

## Résumé
Nous comparons le découpage à taille fixe et le découpage guidé par la
structure du document pour des articles scientifiques en plusieurs langues.
Le découpage structuré conserve les frontières de section et améliore la
précision des citations.

## Méthodes
Le découpage à taille fixe coupe le texte tous les trois cent cinquante
jetons sans tenir compte de la ponctuation. Le découpage structuré détecte
d'abord les titres de section, puis regroupe les phrases jusqu'à atteindre le
budget de jetons, avec une phrase de chevauchement entre segments voisins.

## Résultats
Le découpage structuré a augmenté la précision des citations de soixante-huit
à quatre-vingt-un pour cent. L'ajout d'un fil d'Ariane contenant le titre de
l'article et le nom de la section en tête de chaque segment a apporté quatre
points supplémentaires, car un segment isolé reste interprétable.

## Discussion
L'extraction du texte des fichiers PDF reste la source d'erreur dominante.
Les articles sur deux colonnes produisent des phrases entrelacées lorsque
l'extraction ignore la disposition des blocs.
