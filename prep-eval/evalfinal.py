import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from app.generate_vector import get_embedding  # Your embedding function

# --- Reference Report (same for all models) ---
reference_report = """
Site de la biopsie : Région périanale Indications cliniques : Récidive d’un carcinome Description macroscopique : Nombre de fragments : 1 Taille des fragments : 1.2 cm x 0.7 cm Aspect : Fragment cutané ferme, couleur rosée Description microscopique : Kératinocytes avec anisocaryose, figures de mitose fréquentes. Absence de nécrose ou d’ulcération. Diagnostic : Dysplasie modérée Commentaires : Suivi clinique nécessaire.
"""

# --- Generated Reports from 3 LLMs ---
generated_reports = {
    "mistral": """
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
    """,

    "phi4-mini": """
    **1. Description macroscopique**

Fragment cutané beige, dimensions approximativement 0.4 x 0.3 cm, aspect solide. Lésion perlée persistante visible sur la joue droite.

**2. Description microscopique**

Kératinocytes à hautement atypiques avec anisocaryose et figures de mitose fréquentes. Absence notable d’ulcération ou nécrose. Staining Ber-EP4 positif, indiquant une expression des protéines du réticulum endoplasmique (ER).

**3. Diagnostic**

Carcinome basocellulaire en stade initial.

**4. Conclusion / Commentaires**

En raison de la lésion perlée persistante et de la positivity à Ber-EP4, le diagnostic confirme un carcinome basocellulaire localisé dans la joue droite. Un suivi clinique rigoureux est recommandé pour évaluer tout potentiel de progression ou métastase future. Des études supplémentaires pourraient être nécessaires si des changements sont observés lors du contrôle suivant.
    """,

    "qwen2.5:1.5b": """
    **1. Description macroscopique**

Fragment cutané de 0,4 x 0,3 cm, couleur beige. La biopsie est de la région périaneale et présentait une lésion perlée persistante. 

**2. Description microscopique**

Kératinocytes avec anisocaryose, figures de mitose fréquentes. Absence de nécrose ou d'ulcération.

**3. Diagnostic**

Le diagnostic est de Dysplasie modérée.

**4. Conclusion / Commentaires**

Suite à l'imagerie multidisciplinaire (dont la biopsie macroscopique et microscopique), le patient a été suivi en surveillance clinique. Il a reçu une surveillance régulière pour détecter tout développement ou croissance de la maladie. Les résultats sont en cours d'analyse mais il n'y a actuellement pas de signe émergent qui nécessite une réorientation clinique spécifique.
    """
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
