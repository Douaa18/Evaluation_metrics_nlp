# Métriques d'évaluation en NLP

Exemples de code accompagnant la présentation **« Comment mesurer la qualité d'un modèle de langage »**.
Le notebook `nlp_evaluation_metrics.ipynb` applique les principales métriques d'évaluation sur des exemples courts et reproductibles pour trois tâches :

| Tâche | Métriques |
|---|---|
| Summarization | ROUGE, BERTScore, taux de compression, factualité (NLI) |
| Question Answering | Exact Match, F1 (SQuAD) |
| Text Generation | BLEU, chrF, METEOR, BERTScore, perplexité, distinct-n, self-BLEU |

## Contenu du dépôt

```
.
├── nlp_evaluation_metrics.ipynb   # code et exemples
├── requirements.txt               # dépendances Python
└── README.md                      # cette documentation
```

## Installation et exécution

Prérequis : Python 3.9+ (ou Google Colab, aucune installation locale nécessaire).

```bash
git clone <URL_DU_DEPOT>
cd <NOM_DU_DEPOT>
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook nlp_evaluation_metrics.ipynb
```

Le premier lancement télécharge des modèles (roberta-large pour BERTScore, `facebook/bart-large-mnli`, GPT-2) et des ressources NLTK (`punkt`, `wordnet`, `omw-1.4`). Une connexion Internet est donc nécessaire, et l'exécution est plus rapide avec un GPU (Colab par exemple).

## Documentation du code

Le notebook est découpé en 5 sections. Toutes les fonctions sont écrites à la main quand c'est utile à la compréhension, puis comparées aux implémentations officielles de la bibliothèque `evaluate`.

### 1. Installation et imports
Installe et importe `evaluate`, `nltk`, `pandas`, `numpy`, `matplotlib` et télécharge les ressources NLTK nécessaires à METEOR.

### 2. Summarization
Exemple : un texte source (conseil municipal et budget des transports), un résumé de référence, un **bon résumé** (`candidate_good`) et un résumé **hors sujet** (`candidate_bad`).

| Fonction | Rôle |
|---|---|
| `rouge_scores(pred, ref)` | ROUGE-1, ROUGE-2 et ROUGE-L via `evaluate.load("rouge")` |
| `tokenize`, `ngrams`, `rouge_n(pred, ref, n)` | ROUGE-N recalculé à la main : précision, rappel, F1 sur les n-grammes communs |
| `bert_scores(preds, refs, lang)` | Précision, rappel et F1 sémantiques via BERTScore (embeddings contextuels roberta-large) |
| `compression_ratio(summary, source)` | Nombre de mots du résumé / nombre de mots de la source (mesure la concision, pas la qualité) |
| `faithfulness(source, summary)` | Factualité : probabilités *entailment / neutral / contradiction* avec `facebook/bart-large-mnli` |

Idée : ROUGE compte les mots communs, BERTScore compare le sens, et le NLI vérifie que le résumé n'invente rien (hallucinations).

### 3. Question Answering
Cinq paires question / réponse prédite / réponses acceptées, évaluées à la manière de SQuAD.

| Fonction | Rôle |
|---|---|
| `normalize_answer(s)` | Normalisation SQuAD : minuscules, sans ponctuation, sans articles (a, an, the), espaces nettoyés |
| `exact_match(pred, golds)` | 1 si la réponse normalisée est identique à l'une des références, sinon 0 |
| `f1_single(pred, gold)` / `f1_score(pred, golds)` | F1 au niveau des mots, maximum sur toutes les références |

Les résultats sont rassemblés dans un `DataFrame` (`df_qa`) avec l'EM et le F1 globaux, puis vérifiés avec la métrique officielle `evaluate.load("squad")`.

### 4. Text Generation
Une phrase de référence et trois candidats : **proche**, **reformulée**, **hors sujet**.

| Élément | Rôle |
|---|---|
| BLEU (`sacrebleu`) | Précision de n-grammes (1 à 4) |
| chrF (`chrf`) | N-grammes de caractères, tolère fautes et morphologie |
| METEOR (`meteor`) | Alignement par forme exacte, racine et synonymes (WordNet) |
| BERTScore | Réutilise `bert_scores` pour comparer le sens |
| `perplexity(text)` | Fluidité : exponentielle de la perte de GPT-2 sur le texte (basse = fluide, aucune référence requise) |
| `distinct_n(texts, n)` | Proportion de n-grammes uniques dans un corpus (haut = varié) |
| `self_bleu(texts)` | BLEU de chaque texte contre les autres (bas = varié) |

### 5. Récapitulatif et visualisation
Un tableau (`summary`) liste pour chaque tâche la métrique, ce qu'elle mesure et sa limite. Un diagramme en barres compare les scores normalisés de BLEU, chrF, METEOR et BERTScore pour les trois candidats.

## Limites à garder en tête

- Aucune métrique ne suffit seule : combiner recouvrement, sémantique, fluidité et jugement humain (ou LLM-juge).
- Un bon ROUGE n'exclut pas les hallucinations : ajouter une mesure de factualité.
- En QA, rapporter l'Exact Match et le F1 ensemble.
- Les exemples sont volontairement très courts : les scores illustrent le comportement des métriques, ils ne constituent pas un benchmark.

## Références

- Lin (2004), *ROUGE: A Package for Automatic Evaluation of Summaries*
- Zhang et al. (2020), *BERTScore: Evaluating Text Generation with BERT*
- Rajpurkar et al. (2016), *SQuAD: 100,000+ Questions for Machine Comprehension of Text*
- Papineni et al. (2002), *BLEU* · Popović (2015), *chrF* · Banerjee & Lavie (2005), *METEOR*
- Documentation Hugging Face [`evaluate`](https://huggingface.co/docs/evaluate)
