- python version : 3.11.2

- Lite_CNN-TNN_UT (TensorFlow): 
	* requirements.txt pour environnement virtuel
	  N'hésitez pas à supprimer les librairies qui posent problème lors de l'installation
	* TNN_lite_share.py : entrainer son IA TNN lite (modifier les chemins)
	* convert_tflite.py : convertir le modèle sauvegardé en .tflite
	* prediction_exemple.py : tester des prédictions (utilise prediction.py et convert_A_scan.py)
	* prediction.py : fonction pour réaliser des prédictions (utilise convert_A_scan.py)
	* convert_A_scan.py : fonction pour convertir un A_scan en format utilisable pour l'IA
	* datasets (uniquement test)

- Shakespeare_Gpt (PyTorch): 
	* Data : texte d'entrainement
	* Shakespeare_GPT : petit language model entrainé pour générer du texte de Shakespeare (basé sur l'exemple de Andrej Karpathy https://github.com/karpathy/ng-video-lecture)

- Small_ViT (PyTorch): 
	* Data : Images issues du dataset CIFAR100
	* ViT_Cifar : Petit modèle ViT issue de l'article de référence (https://arxiv.org/abs/2010.11929) 

