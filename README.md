🔎 Problème

La quantification dynamique ne fonctionne que sur les couches Linear et ne supporte pas les convolutions (Conv2d) ou BatchNorm (BatchNorm2d).
Mais ResNet18 utilise des convolutions et BatchNorm, donc le modèle quantifié ne peut pas être exécuté directement.
✅ Solution : Utiliser la quantification statique

La quantification statique fonctionne mieux pour des modèles avec des convolutions (Conv2d).
On doit :

    Préparer le modèle avec torch.quantization.prepare()
    Effectuer une passe de calibration avec des images de test
    Appliquer la quantification finale avec torch.quantization.convert()


Le problème vient du fait que la quantification dynamique (quantize_dynamic) ne fonctionne que sur les couches nn.Linear, et ton modèle ne contient qu'une seule couche Linear (dans self.classifier). Cependant, la quantification dynamique ne modifie pas les convolutions (Conv2d).
Pourquoi il n’y a pas de Linear dans ton modèle quantifié ?

    quantize_dynamic ne quantifie que les couches Linear, LSTM, et GRU.

    Si la couche Linear n'a pas de poids significatifs (ex : elle n'est pas utilisée ou a été fusionnée), elle peut être ignorée.

    Ton modèle a beaucoup de Conv2d, et quantize_dynamic ne les prend pas en charge.
