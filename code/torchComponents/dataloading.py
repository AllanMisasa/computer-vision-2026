import  os
import torch
from torch.utils.data import DataLoader, TensorDataset

'''
Dette er den lidt lettere måde at lave datasæt på end den indbyggede metode i PyTorch.
Det er lettere fordi vi bare ændre i stierne hvis vi vil have andet data.
'''

# Klargør billedlister til binær klassificering
imagesClass1 = []
imagesClass2 = []
labels = []

# Hvilken mappe vores billededata ligger i
rootDirClass1 = "../../images" 
rootDirClass2 = ""

'''
Få relative stier til alle billeder og tilføj til images list.
Selvom det er mere kompliceret en metode end glob eller path som ofte bliver foreslået til denne slags opgave
så foretrækker jeg denne metode med kun det indbyggede os modul. Det er tydeligt hvad hele algoritmen gør,
og vi kan få hele relative stier uden at sætte os ind i et moduls specifikke funktioner.
Repeat for imagesClass2.
'''
for dirPath, subdirs, files in os.walk(rootDirClass1):
    for file in files:
        imagesClass1.append(os.path.join(dirPath, file))

print(imagesClass1)

'''
Udfyld label listen så hvert billede også har et label.
'''

for i in range(len(imagesClass1)):
    labels.append(1) 

'''
Tjek at vi har samme antal billeder og labels 
'''
print((len(imagesClass1)) == (len(labels)))

