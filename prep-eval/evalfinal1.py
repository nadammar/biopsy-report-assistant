import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from app.generate_vector import get_embedding  # Your embedding function

# --- Reference Report (same for all models) ---
reference_report = """
Site de la biopsie : Avant-bras gauche Indications cliniques : Suspicion de mélanome Description macroscopique : Nombre de fragments : 1 Taille des fragments : 0.6 cm x 1.2 cm Aspect : Biopsie pigmentée, contours irréguliers Description microscopique : Hyperplasie de l'épiderme avec atypies marquées. Présence de figures mitotiques. Infiltrat lymphocytaire dense dans le derme. Diagnostic : Carcinome basocellulaire Commentaires : Exérèse complète suggérée."""

# --- Generated Reports from 3 LLMs ---
generated_reports = {
    "mistral": """
    1. Description macroscopique
    - Nombre de fragments: 1
    - Taille des fragments: 0,6 cm x 0,4 cm (non précisé)
    - Aspect: Biopsie pigmentée, contours irréguliers

2. Description microscopique
   - Hyperplasie de l'épiderme avec atypies marquées.
   - Présence de figures mitotiques.
   - Infiltrat lymphocytaire dense dans le derme.

3. Diagnostic
   - Carcinome basocellulaire

4. Conclusion / Commentaires
   - Une exérèse complète est suggérée pour ce carcinome basocellulaire découvert.""",

    "phi4-mini": """
    **1. Description macroscopique**

Nombre de fragments : 2
Taille des fragments : 0,6 cm x 0,4 cm
Aspect : Biopsie pigmentée, contours irréguliers

---

**2. Description microscopique**

Infiltration par des cellules atypiques formant agrès ou figures mitotiques. Présence de mélanocytes à hauteur élevée et d'une lymphocytose marquée dans le derme.

---

**3. Diagnostic : Mélanome malin (maladie de peau) avec potentiel infiltratif en dermis**

---

**4. Conclusion / Commentaires**

L'évolution rapide du pigmentage suggère un stade avancé, nécessitant une étude complémentaire par des marqueurs comme HMB-45 et Melan-A pour confirmer le diagnostic.

---""",

    "qwen2.5:1.5b": """
    1. **Description macroscopique**
   - Nombre de fragments : 1
   - Taille des fragments : 0.6 cm x 1.2 cm
   - Aspect : Biopsie pigmentée, contours irréguliers

2. **Description microscopique**
   - Hyperplasie de l'épiderme avec atypies marquées.
   - Présence de figures mitotiques.
   - Infiltrat lymphocytaire dense dans le derme.

3. **Diagnostic**
   Diagnostic : Carcinome basocellulaire
   Commentaires : Exérèse complète suggérée.

4. **Conclusion / Commentaires**
   Le rapport anatomopathologique clairement démontre la présence d’un mélanome malin sur l’avant-bras gauche, qui a connu une évolution rapide après le diagnostic initial de mélanome. La biopsie colorée à brun foncé et les techniques demandées (HMB-45, Melan-A) ont confirmé cette suspicion potentiellement cancerienne du tissu consulté.

   Le diagnostic final est un carcinome basocellulaire, bien que la description microscopique indique des caractéristiques atypiques de l'épiderme avec les figures mitotiques et une présence d'infiltrat lymphocytaire dense. Bien qu'il n'y ait pas de témoignage clinique spécifique, il semblerait que le diagnostic a été confirmé par les techniques anatomopathologiques nécessaires.

   De plus, étant donné la nature pigmentée de la lésion et son évolution rapide, une exérèse complète serait recommandée pour éliminer toute possibilité d'une métastases ou autres complications potentielles."""
}

# --- BLEU Evaluation ---
def compute_bleu(reference, generated):
    reference_tokens = [reference.split()]
    generated_tokens = generated.split()
    return sentence_bleu(reference_tokens, generated_tokens)

# --- ROUGE Evaluation ---
def compute_rouge(reference, generated):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return scorer.score(reference, generated)

# --- Cosine Similarity Evaluation ---
def compute_cosine_similarity(reference, generated):
    ref_embedding = np.array(get_embedding(reference))
    gen_embedding = np.array(get_embedding(generated))
    ref_embedding = ref_embedding.reshape(1, -1)
    gen_embedding = gen_embedding.reshape(1, -1)
    similarity_score = cosine_similarity(ref_embedding, gen_embedding)
    return similarity_score[0][0]

# --- Evaluation Loop ---
print("\n--- Evaluation Results for Each LLM ---\n")
for model_name, gen_report in generated_reports.items():
    print(f"Model: {model_name}")
    bleu = compute_bleu(reference_report, gen_report)
    rouge = compute_rouge(reference_report, gen_report)
    cosine_sim = compute_cosine_similarity(reference_report, gen_report)

    print(f"  BLEU Score: {round(bleu, 4)}")
    print(f"  ROUGE-1: Precision={round(rouge['rouge1'].precision, 4)}, Recall={round(rouge['rouge1'].recall, 4)}, F1={round(rouge['rouge1'].fmeasure, 4)}")
    print(f"  ROUGE-2: Precision={round(rouge['rouge2'].precision, 4)}, Recall={round(rouge['rouge2'].recall, 4)}, F1={round(rouge['rouge2'].fmeasure, 4)}")
    print(f"  ROUGE-L: Precision={round(rouge['rougeL'].precision, 4)}, Recall={round(rouge['rougeL'].recall, 4)}, F1={round(rouge['rougeL'].fmeasure, 4)}")
    print(f"  Cosine Similarity: {round(cosine_sim, 4)}")
    print("-" * 50)
