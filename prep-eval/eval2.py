import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score

# Load NLTK tokenizers (only needed once)
#import nltk
#nltk.download('punkt')
#nltk.download('wordnet')

# Replace this with your actual embedding function
from app.generate_vector import get_embedding  # Must return a NumPy array


prompt = """
Biopsie de la joue droite, suspicion de carcinome basocellulaire.
Fragment cutané de 0,4 x 0,3 cm, couleur beige.
Présence clinique d’une lésion perlée persistante.
Analyse immunohistochimique demandée : Ber-EP4."""
retrieved_report = """
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
reference_report = retrieved_report  # You can use this or provide a better gold standard

# --- BLEU Score ---
def compute_bleu(reference, generated):
    return sentence_bleu([reference.split()], generated.split())

# --- ROUGE Score ---
def compute_rouge(reference, generated):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return scorer.score(reference, generated)

# --- Cosine Similarity (Semantic) ---
def compute_cosine_similarity(reference, generated):
    ref_emb = get_embedding(reference).reshape(1, -1)
    gen_emb = get_embedding(generated).reshape(1, -1)
    return cosine_similarity(ref_emb, gen_emb)[0][0]

# --- METEOR Score ---
from nltk.tokenize import word_tokenize

def compute_meteor(reference, generated):
    reference_tokens = word_tokenize(reference)
    generated_tokens = word_tokenize(generated)
    return meteor_score([reference_tokens], generated_tokens)


# --- BERTScore ---
def compute_bertscore(reference, generated, lang="fr"):
    P, R, F1 = score([generated], [reference], lang=lang)
    return {
        "precision": P[0].item(),
        "recall": R[0].item(),
        "f1": F1[0].item()
    }

# --- Run All Evaluations ---
def evaluate(reference, generated):
    print("\n--- Evaluation Metrics ---")
    print("BLEU Score:", round(compute_bleu(reference, generated), 4))
    print("METEOR Score:", round(compute_meteor(reference, generated), 4))
    print("Cosine Similarity:", round(compute_cosine_similarity(reference, generated), 4))
    print("ROUGE Scores:")
    rouge = compute_rouge(reference, generated)
    for k, v in rouge.items():
        print(f"  {k}: P={v.precision:.4f}, R={v.recall:.4f}, F1={v.fmeasure:.4f}")
    bert = compute_bertscore(reference, generated)
    print("BERTScore:")
    for k, v in bert.items():
        print(f"  {k}: {v:.4f}")

# --- Main ---
if __name__ == "__main__":
    evaluate(reference_report, generated_report)
