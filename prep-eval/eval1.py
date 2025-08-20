import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from app.generate_vector import get_embedding  # Your custom embedding function

# --- Reference and Generated Reports ---
reference_report1 = """
Site de la biopsie : Région périanale Indications cliniques : Récidive d’un carcinome Description macroscopique : Nombre de fragments : 1 Taille des fragments : 1.2 cm x 0.7 cm Aspect : Fragment cutané ferme, couleur rosée Description microscopique : Kératinocytes avec anisocaryose, figures de mitose fréquentes. Absence de nécrose ou d’ulcération. Diagnostic : Dysplasie modérée Commentaires : Suivi clinique nécessaire.
"""

generated_report = """
1. Description macroscopique
     - Nombre de fragments: 1
     - Taille des fragments: 0,4 x 0,3 cm
     - Aspect: Fragment cutané dur, couleur beige

2. Description microscopique
    - Kératinocytes avec anisocaryose, figures de mitose fréquentes. Absence de nécrose ou d’ulcération.

3. Diagnostic
    - Dysplasie modérée, compatible avec un carcinome basocellulaire

4. Conclusion / Commentaires
   - Suivi clinique et éventuelle biopsie pérenne sont nécessaires pour confirmer le diagnostic et évaluer la réponse au traitement. L'analyse immunohistochimique Ber-EP4 sera utile pour renforcer ce diagnostic.
"""

# --- BLEU Evaluation ---
def compute_bleu(reference, generated):
    reference_tokens = [reference.split()]
    generated_tokens = generated.split()
    return sentence_bleu(reference_tokens, generated_tokens)

# --- ROUGE Evaluation ---
def compute_rouge(reference, generated):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return scores

# --- Cosine Similarity Evaluation (fixed) ---
def compute_cosine_similarity(reference, generated):
    # Convert to NumPy arrays
    ref_embedding = np.array(get_embedding(reference))
    gen_embedding = np.array(get_embedding(generated))

    # Reshape for sklearn
    ref_embedding = ref_embedding.reshape(1, -1)
    gen_embedding = gen_embedding.reshape(1, -1)

    # Compute similarity
    similarity_score = cosine_similarity(ref_embedding, gen_embedding)
    return similarity_score[0][0]

# --- Run Evaluation ---
bleu_score = compute_bleu(reference_report1, generated_report)
rouge_scores = compute_rouge(reference_report1, generated_report)
cosine_sim = compute_cosine_similarity(reference_report1, generated_report)

# --- Display Results ---
print("\n--- Evaluation Metrics ---")
print("BLEU Score:", round(bleu_score, 4))
print("ROUGE Scores:")
for key, value in rouge_scores.items():
    print(f"  {key}: Precision={value.precision:.4f}, Recall={value.recall:.4f}, F1={value.fmeasure:.4f}")
print("Cosine Similarity Score:", round(cosine_sim, 4))
