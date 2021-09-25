import unittest
from kdmt.text import split_text_to_sentences
text_original = "Je suis l'administratrice de M. Gungor Ugur, adresse d'habitat Rue Frans Pepermans 41, 1140 Evere. Vous trouverez ma désignation en annexe"

tokenized_text = ["Je suis l'administratrice de M. Gungor Ugur, adresse d'habitat Rue Frans Pepermans 41, 1140 Evere.", "Vous trouverez ma désignation en annexe"]

text_original2="s'il vous plaît répondez-moi sur ma question datant du 01/11/2018. Votre ref. 8555738808  Merci d'avance."
tokenized_text2=["s'il vous plaît répondez-moi sur ma question datant du 01/11/2018.", "Votre ref. 8555738808  Merci d'avance."]

text_original3="Confirmation de réception de votre dossier avec réf. 8536219237 Sauf que je n'ai malheureusement pas reçu une réponse à l'objet de ma plainte."
tokenized_text3=["Confirmation de réception de votre dossier avec réf. 8536219237 Sauf que je n'ai malheureusement pas reçu une réponse à l'objet de ma plainte."]


text_original4="Vous trouverez le document d'acquisition en pièce jointe. Pouvons-nous vous demander de bien vouloir faire le nécessaire"
tokenized_text4=["Vous trouverez le document d'acquisition en pièce jointe.", "Pouvons-nous vous demander de bien vouloir faire le nécessaire"]

def areEqual(arr1, arr2):
    # Linearly compare elements
    for i in range(0, len(arr1) - 1):
        if (arr1[i] != arr2[i]):
            return False
    return True

class TestKolibriTokeniers(unittest.TestCase):

    def test_split_sentence(self):
        splited=split_text_to_sentences(text=text_original, multi_line=False)
        assert areEqual(splited, tokenized_text)

        splited=split_text_to_sentences(text=text_original2, multi_line=False)
        assert areEqual(splited, tokenized_text2)


        splited=split_text_to_sentences(text=text_original3, multi_line=False)
        assert areEqual(splited, tokenized_text3)

        splited=split_text_to_sentences(text=text_original4, multi_line=False)
        assert areEqual(splited, tokenized_text4)

