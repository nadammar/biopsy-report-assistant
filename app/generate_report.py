import ollama

def generate_report(prompt, similar_contents, patient_info):
    context = "\n\n".join(similar_contents)

    patient_header = (
        "RAPPORT D'ANATOMIE PATHOLOGIQUE\n\n"
        "Identification du patient:\n"
        f"Nom : {patient_info['patient_name']}\n"
        f"Date de naissance : {patient_info['birth_date']}\n"
        f"Numéro de dossier : {patient_info['dossier_number']}\n\n"
        "Informations cliniques:\n"
        f"Médecin traitant : {patient_info['doctor_name']}\n"
        f"Date de la biopsie : {patient_info['biopsy_date']}\n"
    )

    llm_prompt = (
    "Tu es un assistant médical expert en anatomopathologie.\n\n"
    "Ton objectif est de générer un rapport d’anatomie pathologique structuré en français médical, en te basant uniquement sur :\n"
    "- Le contexte clinique du médecin pour extraire tous les détails pertinents (site, taille, aspect, techniques spéciales, etc.) utiles aux sections “Description macroscopique” et “Description microscopique”.\n"
    "- Le contenu des rapports similaires pour écrire les sections “Diagnostic” et “Conclusion / Commentaires”. Résume et adapte leur contenu pour formuler un diagnostic cohérent avec le contexte clinique. N’utilise pas de copier-coller.\n\n"
    "Tu dois générer les 4 sections demandées dans tous les cas, même si certaines informations sont absentes. Dans ce cas, indique « Non précisé ».\n\n"
    "N’écris aucune section d’identification du patient ni d’informations cliniques — elles sont déjà fournies. Évite toute répétition.\n\n"
    "Rédige uniquement les 4 sections suivantes dans cet ordre exact :\n"
    "1. Description macroscopique\n"
    "2. Description microscopique\n"
    "3. Diagnostic\n"
    "4. Conclusion / Commentaires\n\n"
    "Formate clairement chaque section avec un titre en gras suivi de son contenu. Pas de phrases introductives ou stylistiques. Ne répète pas l’indication clinique dans les sections. Ne termine pas par une phrase générique sauf si elle est justifiée.\n\n"
    f"Contexte clinique du médecin (à ne pas inclure dans le rapport) :\n{prompt}\n\n"
    f"Rapports similaires :\n{context}\n"
)


    try:
        response = ollama.chat(
            model="mistral", #phi4-mini , qwen2.5:1.5b ,mistral 
            messages=[
                {"role": "user", "content": llm_prompt}
            ]
        )
        return patient_header + "\n" + response['message']['content']
    except Exception as e:
        return f"❌ Error generating report: {str(e)}"
