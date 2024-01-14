from pydub import AudioSegment

def mixer_sons(son1_path, son2_path, output_path):
    # Charger les fichiers audio
    son1 = AudioSegment.from_mp3(son1_path)
    son2 = AudioSegment.from_mp3(son2_path)

    # Ajuster la durée des fichiers audio pour qu'ils aient la même longueur
    min_length = min(len(son1), len(son2))
    son1 = son1[:min_length]
    son2 = son2[:min_length]

    # Mélanger les deux fichiers audio
    son_melange = son1.overlay(son2)

    # Enregistrer le fichier audio résultant
    son_melange.export(output_path, format="mp3")

# Exemple d'utilisation
son1_path = "NotesPiano_-_04122023_19.24.mp3"
son2_path = "AMBSea_Mer_vagues_moyennes_et_mouettes_ID_0267_LS.mp3"
output_path = "01.mp3"

mixer_sons(son1_path, son2_path, output_path)