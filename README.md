🔎 Problème

La quantification dynamique ne fonctionne que sur les couches Linear et ne supporte pas les convolutions (Conv2d) ou BatchNorm (BatchNorm2d).
Mais ResNet18 utilise des convolutions et BatchNorm, donc le modèle quantifié ne peut pas être exécuté directement.
✅ Solution : Utiliser la quantification statique

La quantification statique fonctionne mieux pour des modèles avec des convolutions (Conv2d).
On doit :

    Préparer le modèle avec torch.quantization.prepare()
    Effectuer une passe de calibration avec des images de test
    Appliquer la quantification finale avec torch.quantization.convert()
