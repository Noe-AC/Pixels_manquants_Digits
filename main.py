import numpy as np
import random

import timeit # Pour avoir la durée de calcul d'un calcul
import math

import numpy as np
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import pandas as pd
import itertools

import matplotlib.pylab as plt
from PIL import Image # pour convertir un array (64,0) à une image (8,8)



####################################################

def loi_binomiale(n,x,p): # P(X=x)
	return st.binom.pmf(k=x,n=n,p=p)

####################################################
"""
On suppose qu'il y a une probabilité qu'un pixel d'une image soit connu (et q=1-p qu'il soit inconnu).
La loi binomiale calcule la probabilité qu'il y ait x « bons » pixels dans les n=64=8x8 pixels d'une image de 8x8 pixels.
"""
p = 0.8 # probabilité de bon pixel
q = 1-p # probabilité de mauvais pixel
# on dira qu'un pixel est bon si on a 1 et mauvais si on a 0
population = [0,1]
weights = [q,p]
#thing = np.random.choice(population, p=weights)

"""
nb_good_pixels = 0
for i in range(100):
	good_or_not = np.random.choice(population, p=weights)
	if good_or_not:
		nb_good_pixels += 1
print(nb_good_pixels)
"""
# Je dois calculer ça pour : 1700 images, 64 pixels, ça fait 1700*64=108800 approx. 100000 calculs :
"""
nb_good_pixels = 0
for i in range(100000):
	good_or_not = np.random.choice(population, p=weights)
	if good_or_not:
		nb_good_pixels += 1
print(nb_good_pixels)
"""

# Une fonction qui donne un array population [0,1,2,...,n] et l'autre les probabilités [P(X=0),P(X=1),...,P(X=n)] qu'il y ait exactement x bons pixels dans une image
def population_and_weights_binomial_law(n,p):
	population_nb_pixels = list(range(n+1)) # variable aléatoire = nombre de pixels corrects
	weights_nb_pixels = []
	for nb_pixels in population_nb_pixels:
		prob = loi_binomiale(n=64,x=nb_pixels,p=p)
		weights_nb_pixels.append(prob)
	return population_nb_pixels,weights_nb_pixels

# On crée alors deux array pour la loi binomiale B(n,p), un est la population [0,1,...,64] l'autre est les probabilités [P(X=0),P(X=1),...,P(X=n)]
population_nb_pixels,weights_nb_pixels = population_and_weights_binomial_law(n=64,p=p)


#print(random.sample(range(100000000), 3))
"""
for i in range(100):
	nb_good_pixels = np.random.choice(population_nb_pixels, p=weights_nb_pixels)
	good_pixels = random.sample(range(64), nb_good_pixels)
	print(nb_good_pixels,good_pixels)
"""

####################################################

# Ici on prépare les données

# On importe les données
digits = datasets.load_digits()
X = digits.data   # un array numpy de taille (1797, 64)
y = digits.target # un array numpy de longueur 1797

# On normalise les pixels de X à des float en [0,1] :
m,n = X.shape # m=1797, n=64
id_number = np.linspace(0,m-1,m) # les id sont un array numpy [0,1,...,m-1]
X_normed = X/16 # array numpy de pixels normalisés en [0.0000,1.0000]

# On split les données en train+test
test_size=0.25
random_state = 27
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(X_normed,y,id_number,test_size=test_size, random_state=random_state)
#print(X_train.shape) # (1347, 64)
#print(X_test.shape)  # (450, 64)


# On unie (y_train,X_train) dans un seul array Xy_train
Xy_train = np.zeros(( len(y_train) , len(X_train[0])+1 ))
Xy_train[:,0]  = y_train
Xy_train[:,1:] = X_train

# On extrait les types de chiffres
Xy_train_df = pd.DataFrame(Xy_train)
X_train_filtered = []
y_train_filtered = []
chiffres = list(range(10))
for chiffre in chiffres:
	indices = (Xy_train_df[0]==chiffre)
	Xy_train_df_chiffre = Xy_train_df[indices] # un dataframe pandas par chiffre pour Xy_train
	Xy_train_chiffre = Xy_train_df_chiffre.to_numpy() # un array numpy par chiffre pour Xy_train
	X_train_chiffre = Xy_train_chiffre[:,1:] # un array numpy par chiffre pour X_train
	y_train_chiffre = Xy_train_chiffre[:,0] # un array numpy par chiffre pour y_train
	X_train_filtered.append(X_train_chiffre)
	y_train_filtered.append(y_train_chiffre)



# On remet les données X_train et y_train en ordre :
X_train = np.vstack(X_train_filtered)
y_train = list(itertools.chain.from_iterable(y_train_filtered))


# On regarde il y a combien d'images par chiffre d'entraînement
nb_par_chiffre = [] # nombre d'images par chiffre
for chiffre in chiffres:nb_par_chiffre.append(len(X_train_filtered[chiffre]))
print("Nombre d'images par chiffre : ",nb_par_chiffre) # [139, 145, 134, 137, 131, 133, 129, 141, 129, 129]



# Avant de perforer les données, on se fait un backup non perforé
X_train_filtered_original = []
for chiffre in chiffres:
	X_train_filtered_original.append(np.array(X_train_filtered[chiffre]))
	# On doit re-caster l'array en np.array sinon les modifications sur X_train_filtered vont affecter le backup
X_test_original = np.array(X_test)



# Fonction de perforation
def perforation_images(X,p):
	m,n = X.shape # X doit être un array numpy de m lignes et n colonnes
	population_nb_pixels,weights_nb_pixels = population_and_weights_binomial_law(n=n,p=p)
	for i in range(m):
		nb_good_pixels = np.random.choice(population_nb_pixels, p=weights_nb_pixels) # le nombre de bons pixels est une variable aléatoire
		nb_bad_pixels = n - nb_good_pixels # le nombre de mauvais pixels est une variable aléatoire
		bad_pixels = random.sample(range(n), nb_bad_pixels) # on choisir aléatoirement les mauvais pixels
		for pixels in bad_pixels:
			X[i][pixels] = -1
	return X
def perforation_images_filtered(X_filtered,p):
	chiffres = list(range(10))
	for chiffre in chiffres:
		X_filtered[chiffre] = perforation_images(X=X_filtered[chiffre],p=p)
	return X_filtered



####################################################


# Cette fonction prend un array numpy de taille (m,n) de m images de n pixels présents ou non
# Ça renvoie la distance normalisée entre chaque lignes
# i.e. ça renvoie un array numpy de taille (m,m)
def dist_images_images(X): # X est supposé être un array numpy de taille (m,n)
	if type(X)!=np.ndarray:
		print("Erreur - dist_images_images - X pas du bon type")
		return -1
	if len(X.shape)!=2:
		print("Erreur - dist_images_images - X pas de la bonne dimension")
		return -1
	m,n = X.shape
	# On reshape X :
	Xi = X.reshape(m,1,n) # taille (m,1,n)
	Xj = X.reshape(1,m,n) # taille (1,m,n)
	# On regarde quels pixels sont présents
	Xi_bool  = (Xi!=-1) # taille (m,1,n)
	Xj_bool  = (Xj!=-1) # taille (1,m,n)
	Xij_bool = Xi_bool*Xj_bool # taille (m,m,n)
	nombres  = np.count_nonzero(Xij_bool,axis=2) # taille (m,m) nombre de pixels communs
	#print(nombres)
	d2 = np.sum( (Xij_bool*(Xi-Xj))**2 , axis=2) # taille (m,m) distances entre images i et j
	#print(d2)
	# Là où d2 et nombres sont nuls en même temps, on met les distances à 64 :
	d2[(d2==0.0)*(nombres==0)] = 64.0
	nombres[nombres==0] = 1
	# On normalise les distances
	d2_normed = d2*64.0/nombres # taille (m,m), distances normalisées entre images i et j
	# Ici on donne plus de poids aux images plus près.
	# C'est optionnel.
	# Si ça n'aide pas à la reconstruction d'images je vais le mettre à off
	d2_normed /= nombres 
	# On met la distance maximale 64.0 là où c'est infini
	d2_normed[d2_normed==np.inf] = 64.0 # on met la distance maximale là où c'est infini
	# Enfin, ici on veut la distance entre des images différentes
	# Donc je vais mettre la diagonale à 64.0, i.e. la distance de l'image i avec elle-même à 64.0 (et non 0)
	d2_normed += 64.0*np.eye(m) # on ajoute une matrice diagonale (m,m) avec 64.0 sur la diagonale
	return d2_normed

# Cette fonction calcule la distance d'une image aux autres images
# L'indice i est la position de l'image x dans l'array d'images X
def update_dist_image_images(X,x,i):
	if type(X)!=np.ndarray:
		print("Erreur - update_dist_image_images - X pas du bon type")
		return -1
	if len(X.shape)!=2:
		print("Erreur - update_dist_image_images - X pas de la bonne dimension")
		return -1
	m,n=X.shape
	if len(x)!=n:
		print("Erreur - update_dist_image_images - x pas de la bonne dimension")
		return -1
	X_bool = (X!=-1) # taille (m,n), bool des pixels existants des images de X
	x_bool = (x!=-1) # taille (m), bool des pixels existants de l'image x
	Xx_bool = X_bool*x_bool # taille (m,n), bool des pixels existants communs entre x et les images de X
	nombres = np.count_nonzero(Xx_bool,axis=1) # de taille (m), nombre de pixels présents dans chaque image
	d2 = np.sum( (Xx_bool*(X-x))**2, axis=1 ) # taille (m), distances entre x et les images de X
	# Là où d2 et nombres sont nuls en même temps, on met les distances à 64 :
	d2[(d2==0)*(nombres==0)] = 64.0
	nombres[nombres==0] = 1
	# On normalise d2 :
	d2_normed = d2*64.0/nombres # taille (m), distances normalisées entre l'image x et celles de X
	# Ici on donne plus de poids aux images plus près.
	# C'est optionnel.
	# Si ça n'aide pas à la reconstruction d'images je vais le mettre à off
	d2_normed /= nombres 	
	# On met la distance maximale 64.0 là où c'est infini
	#d2_normed[d2_normed==np.inf] = 64.0 # on met la distance maximale là où c'est infini
	# Enfin on met la distance maximal pour x envers elle-même
	d2_normed[i] = 64.0
	return d2_normed




# Ici le but est de reconstruire les images de X_train
def reconstruction_images_chiffre(X_train_chiffre):
	images_bool = (X_train_chiffre!=-1) # taille (139, 64), un bool de pixels corrects par image
	nb_pixels = np.count_nonzero(images_bool,axis=1) # taille (139), nombre de pixels corrects par image
	m,n = images_bool.shape # m=139, n=64
	images_bool_i = images_bool.reshape(m,1,n) # taille (139,1,64)
	images_bool_j = images_bool.reshape(1,m,n) # taille (1,139,64)
	# On peut regarder une matrice 139x139 qui dit la distance entre chaque paire d'images :
	dist_XX = dist_images_images(X_train_chiffre) # taille (139,139), distance entre les paires d'images
	# On peut regarder trois array bool de taille (139,139,64)
	images_bool_i_and_j = images_bool_i & images_bool_j # taille (139,139,64), pixels communs aux deux, & est "et"
	images_bool_j_but_not_i = images_bool_i < images_bool_j # taille (139,139,64), pixels en j mais pas en i
	# On peut aussi regarder le nombre d'images ayant un nombre donné de pixels
	nb_images_par_nb_pixels = np.zeros(n+1,dtype=int) # une liste qui contient le nombre d'images ayant le nombre de pixels donnés
	for i in range(n+1):nb_images_par_nb_pixels[i]=len(nb_pixels[nb_pixels==i]) # clé = nombre de pixels, valeur = nombre d'images
	#print("Nb. d'im. / (nb. pixels) : ",list(nb_images_par_nb_pixels[35:65]))
	#print("nb_images_par_nb_pixels =",nb_images_par_nb_pixels)
	# Maintenant on commence l'algo de reconstruction d'images
	# On doit :
	# - minimiser la distance entre deux points
	# - maximiser le nombre de pixels communs
	# - s'assurer d'avoir des pixels à transférer
	# Ok. J'ai modifié la fonction de distance de sorte que ça prenne en compte le nombre de pixels communs entre les images
	# Plus il y a de pixels communs, plus la distance modifiée est petite de sorte qu'il y a une plus grande chance que les deux images soient "près".

	# On met les images de départ dans des arrays de reconstruction
	X_train_chiffre_reconstructed = X_train_chiffre # taille (139,64)

	# Ici c'est le début de la reconstruction.
	# Il faudra que je mette ça dans une boucle

	for k in range(m*m*n):
		# Début de la boucle de reconstruction
		# On commence par les images qui ont le plus de pixels.
		# On regarde le nombre de pixels maximal que possèdent les images
		nb_pixels_max = (np.where(nb_images_par_nb_pixels!=0))[0][-1] # c'est le dernier élément de la liste des indices de la liste des "nb d'image par nb. de pixels"
		if nb_pixels_max==64:
			if len((np.where(nb_images_par_nb_pixels!=0))[0])==1:
				print("reconstruction done")
				break
			else:
				nb_pixels_max = (np.where(nb_images_par_nb_pixels!=0))[0][-2]
		#print("\nNombre de pixels max = ",nb_pixels_max)
		nb_dimages = nb_images_par_nb_pixels[nb_pixels_max] # On regarde le nombre d'images ayant ce nombre de pixels
		#print("Nombre d'images avec pixels max = ",nb_dimages)
		images_max_pix = np.where(nb_pixels==nb_pixels_max)[0] # On regarde quelles images ont ce nombre de pixels
		#print("Images avec nombre maximal de pixels = ",images_max_pix)
		images_max_pix_first = images_max_pix[0] # Là on s'intéresse à la première image ayant le nombre de pixels max
		#print("Première image avec nombre maximal de pixels = ",images_max_pix_first)
		# On prend cette image et on regarde l'autre image qui est la plus près selon la fonction de distance modifiée
		# On a déjà calculé les distances entre les images i et j
		i = images_max_pix_first # Ici j'ai l'indice i
		distances = dist_XX[i] # On  prend la matrices des distances à la ligne i et on regarde l'indice du min de cette ligne
		dist_min = np.amin(distances) # On regarde la distance minimale
		#print("Distance minimale = ",dist_min)
		images_min_dist = np.where(distances==dist_min)[0] # On regarde les images qui sont à distance minimale
		#print("Images avec distance minimale = ",images_min_dist)
		images_min_dist_first = images_min_dist[0] # On prend juste la première image à distance minimale
		j = images_min_dist_first # On a l'indice j
		#print("La distance minimale est = ",distances[j])
		#print("Nombre de pixels de la première image = ",np.count_nonzero(images_bool[i]))
		#print("Nombre de pixels de la seconde image = ",np.count_nonzero(images_bool[j]))
		#print("Nombre de pixels communs = ",np.count_nonzero(images_bool_i_and_j[i,j]))
		nombre_pixels_transferables = np.count_nonzero(images_bool_j_but_not_i[i,j])
		#print("Nombre de pixels transférables de j à i = ",nombre_pixels_transferables)
		# S'il n'y a aucun pixel transférable on met une distance 64.0 entre i et j puis on recommence la boucle
		if nombre_pixels_transferables==0:
			dist_XX[i,j] = 64.0
			#print("Aucun pixel transférable, on recommence la boucle")
			continue

		# On transfert les pixels de l'image j en l'image i car l'image j était la plus près de l'image i
		# D'abord on sait quels pixels sont à transférer 
		pixels_transferables = np.where(images_bool_j_but_not_i[i,j]==True)[0]
		#print("Pixels transférables",pixels_transferables)
		# On transfert les pixels de l'image j à l'image i
		image_i = X_train_chiffre[i] # On prend l'image i
		image_j = X_train_chiffre[j] # On prend l'image j
		image_i[pixels_transferables] = image_j[pixels_transferables] # On transfert les pixels de image_j à image_i
		X_train_chiffre_reconstructed[i] = image_i # On met l'image i reconstruite dans X_train reconstruit
		image_i_bool = (image_i!=-1)
		image_j_bool = (image_j!=-1)
		image_i_nb_pixels = np.count_nonzero(image_i_bool)
		#print("Nouveau nombre de pixels pour l'image i = ",image_i_nb_pixels)
		# Maintenant qu'on a transféré les pixels dans l'image i il faut mettre à jour plusieurs choses.
		# On met à jour image_bool, taille (139,64), contenant les pixels corrects par image
		images_bool[i] = image_i_bool # on met le nouveau nombre de pixels corrects pour l'image i
		# On met à jour nb_pixels, taille (139), contenant le nombre de pixels corrects par image
		nb_pixels[i] = image_i_nb_pixels # on met à jour le nombre de pixels corrects pour l'image i
		images_bool_i = images_bool.reshape(m,1,n) # taille (139,1,64)
		images_bool_j = images_bool.reshape(1,m,n) # taille (1,139,64)
		images_bool_i_and_j = images_bool_i & images_bool_j # taille (139,139,64), pixels communs aux deux, & est "et"
		images_bool_j_but_not_i = images_bool_i < images_bool_j # taille (139,139,64), pixels en j mais pas en i
		# L'ancien nombre de pixels est nb_pixels_max
		# Le nouveau nombre de pixels est image_i_nb_pixels
		nb_images_par_nb_pixels[nb_pixels_max] -= 1
		nb_images_par_nb_pixels[image_i_nb_pixels] += 1
		#print("Nouveau nb. d'im. / (nb. pixels) : ",list(nb_images_par_nb_pixels[35:65]))
		# Maintenant on calcule la distance de la nouvelle image aux images reconstruite
		# i.e. on calcule la distance entre image_i et X_train_chiffre_reconstructed :
		nouvelles_distances = update_dist_image_images(X=X_train_chiffre_reconstructed,x=image_i,i=i)
		dist_XX[i,:] = nouvelles_distances # On met les nouvelles distances horizontalement
		dist_XX[:,i] = nouvelles_distances # On met les nouvelles distances verticalement
		# Puis on s'assure que l'image i et l'image j ne soient plus près l'un de l'autre au moins jusqu'à ce qu'une des deux images soit mise à jour :
		dist_XX[i,j] = 64.0
		dist_XX[j,i] = 64.0
	# Fin de la boucle de reconstruction
	return X_train_chiffre_reconstructed


def reconstruction_images_filtered(X_train_filtered):
	chiffres = list(range(len(X_train_filtered)))
	for chiffre in chiffres:
		print("Reconstruction du chiffre : ",chiffre)
		X_train_filtered[chiffre] = reconstruction_images_chiffre(X_train_chiffre=X_train_filtered[chiffre])
	return X_train_filtered


def remplissage_blanc_images(X):
	return X*(X!=-1) # on met les pixels absents (-1) à un pixel blanc (0)
def remplissage_blanc_images_filtered(X_filtered):
	chiffres = list(range(10))
	for chiffre in chiffres:
		X_train_filtered[chiffre] = remplissage_blanc_images(X=X_train_filtered[chiffre])
	return X_train_filtered

def remplissage_aleatoire_images(X):
	m,n = X.shape
	X_bool_presents = (X!=-1) # taille (m,n) qui est True là où le pixel existe
	X_bool_absents  = (X==-1) # taille (m,n) qui est True là où le pixel est absent
	X = X*X_bool_presents # on met les pixels absents (-1) à un pixel blanc (0)
	pixels_aleatoires = np.random.random((m,n)) # taille (m,n) de nombres aléatoires en [0.0, 1.0[
	X = X + pixels_aleatoires*X_bool_absents # on met les nouveaux pixels blancs (0) à des pixels aléatoires
	return X
def remplissage_aleatoire_images_filtered(X_filtered):
	chiffres = list(range(10))
	for chiffre in chiffres:
		X_train_filtered[chiffre] = remplissage_aleatoire_images(X=X_train_filtered[chiffre])
	return X_train_filtered		

# Cette fonction reconstruit une image comme suit :
# un pixel manquant est remplacé par la valeur moyenne du pixel sur les autres images
# Cette fonction n'a de sens que sur X_train_chiffre car là les images se ressemblent, mais celles de X_test ne sont pas filtrées
# En pratique, on peut l'utiliser sur X_test mais ça ne sert à rien, mieux vaut faire du remplissage aléatoire dans X_test
def remplissage_moyenne_images_chiffre(X):
	m,n = X.shape
	X_bool_presents = (X!=-1) # taille (m,n) qui est True là où le pixel existe
	X_bool_absents  = (X==-1) # taille (m,n) qui est True là où le pixel est absent
	nombres = np.count_nonzero(X_bool_presents,axis=0) # taille (n), nombre d'images ayant un pixel donné
	X = X*X_bool_presents # on met les pixels absents (-1) à un pixel blanc (0)
	X_sum = np.sum(X,axis=0) # taille (n) on somme la valeur des pixels sur chaque images
	# Avant de faire la moyenne on s'assure de ne pas avoir une division 0/0
	pixels_problematiques = (X_sum==0)*(nombres==0) # On regarde là où la somme est nulle et il n'y a aucun pixel
	# Là où on a 0/0 on va mettre un pixel blanc (0)
	nombres[pixels_problematiques] = 1
	X_moy = X_sum/nombres # taille (n), on fait la moyenne
	# On crée une table (m,n) où chaque ligne est X_moy
	indice_i = np.ones((m,1))
	indice_j = X_moy.reshape(1,n)
	X_moy_table = indice_i*indice_j # taille (m,n)
	# Maintenant là où on a déjà un pixel on met la valeur moyenne à 0
	X_moy_table = X_bool_absents*X_moy_table # taille (m,n), val. moy. là où on n'avait pas de pixel
	# Maintenant on met les pixels moyens dans les images là où on en a besoin
	X = X + X_moy_table
	return X
def remplissage_moyenne_images_filtered(X_filtered):
	chiffres = list(range(10))
	for chiffre in chiffres:
		X_filtered[chiffre] = remplissage_moyenne_images_chiffre(X=X_filtered[chiffre])
	return X_filtered



"""
Les blocs d'images filtrées X_train_chiffre_bool ont dimensions :
shape = (139, 64)
shape = (145, 64)
shape = (134, 64)
shape = (137, 64)
shape = (131, 64)
shape = (133, 64)
shape = (129, 64)
shape = (141, 64)
shape = (129, 64)
shape = (129, 64)
"""





####################################################

# Ici on fait de l'apprentissage machine entre X_train et X_test

# D'abord, je construit mon propre algorithme KNN
def distance_normalisee(data1,data2):
	if type(data1)!=np.ndarray or type(data2)!=np.ndarray:
		print("Erreur - distance_normalisée - data1 ou data2 pas du bon type")
		return -1
	data_bool_mix = (data1!=-1)*(data2!=-1) # on regarde les pixels présents en commun dans les deux images
	nombre = np.count_nonzero(data_bool_mix)
	d2 = np.sum( (data_bool_mix*(data1-data2))**2 )
	if nombre==0:
		return 64.0 # distance maximale entre deux vecteurs en [0,1]^64
	else:
		return d2*64.0/nombre # on retourne la norme normalisée selon le nombre de pixels présents


def closest_point(X,x): # ici on a un array X de vecteurs et un point x. Ça donne l'indice i de X t.q. X[i] est le plus proche de x
	indice = 0
	d2_min = 64.0 # la distance maximale est 64
	for i in range(len(X)):
		d2 = distance_normalisee(data1=X[i],data2=x)
		if d2<d2_min:
			d2_min=d2
			indice=i
	if d2_min==64:
		print("Attention, tous les points ont distance 64.0")
	return indice


def closest_point_optim(X,x): # ici c'est comme closest_point mais optimisé
	indice = 0
	d2_min = 64.0 # la distance maximale est 64
	X_bool = (X!=-1)
	x_bool = (x!=-1)
	Xx_bool = X_bool*x_bool # les pixels communs entre X et x sous forme d'array de la forme de X
	Xx_dist = (Xx_bool*(X-x))**2 
	distances = np.sum( Xx_dist , axis=1)
	nombres = np.count_nonzero(Xx_bool,axis=1)
	distances *= 64.0/nombres # on normalise les distances
	d2_min = np.amin(distances)
	result = np.where(distances == d2_min)
	resultats = result[0]
	if len(resultats)==0:
		#print("Attention, tous les points ont distance 64.0")
		return 0
	else:
		indice = resultats[0]
		return indice

def KNN(y_train,X_train,X_test):
	#print("KNN - Nombre d'images à classifier : ",len(X_test))
	y_pred = []
	for i in range(len(X_test)):
		indice = closest_point_optim(X=X_train,x=X_test[i])
		y_pred.append(y_train[indice])
	return y_pred


"""
# Ici je voulais tout faire en Broadcast Numpy mais j'ai dû faire une erreur
# Le calcul n'est pas fiable quand il manque des pixels
# Aussi il prend 0.7743367989999999 secondes à exécuter alors que l'autre fonction KNN prend juste 0.36967324
# Bref, je vais rester avec l'autre fonction qui est plus rapide et plus fiable
def KNN_optim(y_train,X_train,X_test):
	nb_pixels    = len(X_train[0]) # 64 pixels
	len_test     = len(X_test) # 450 images
	len_train    = len(X_train) # 1347 images
	X_test_bool  = (X_test!=-1) # taille ( 450,64) # sera l'indice i
	X_train_bool = (X_train!=-1) # taille (1347,64) # sera l'indice j
	X_test_bool  = X_test_bool.reshape(len_test,1,nb_pixels) # taille (450,1,64)
	X_train_bool = X_train_bool.reshape(1,len_train,nb_pixels) # taille (1,1347,64)
	X_tt_bool    = X_test_bool*X_train_bool # taille (450,1347,64) contient les pixels communs en bool
	X_test       = X_test.reshape(len_test,1,nb_pixels) # taille (450,1347,64)
	X_train      = X_train.reshape(1,len_train,nb_pixels) # taille (450,1347,64)
	distances    = np.sum((X_train_bool*(X_test-X_train))**2,axis=2) # taille (450,1347), on somme sur les pixels
	nombres      = np.count_nonzero(X_tt_bool,axis=2) # taille (450,1347), nombre de pixels communs entre les paires d'images
	distances    = distances*nb_pixels/nombres # taille (450,1347), on normalise
	d2_min       = np.amin(distances,axis=1) # taille (450), dist. min. du point de X_test au nuage X_train
	y_pred       = np.zeros(len_test) # longueur 450
	for i in range(len_test): # longueur 450
		for j in range(len_train): # longueur 1347
			if distances[i,j]==d2_min[i]:
				y_pred[i] = y_train[j]
				break
	return y_pred
	#d2_min       = d2_min.reshape(len_test,1) # taille (450,1)
	#d2_indices   = distances==d2_min # taille (450,1347) booléen où la distance est minimale. Une ligne peut avoir aucun True si il n'y avait aucun pixels en communs
	#result = np.where(distances == d2_min)
	# Bon la double boucle pourrait probablement être optimisée en termes de broadcast


start = timeit.default_timer()
y_pred = KNN_optim(y_train=y_train,X_train=X_train,X_test=X_test)
stop = timeit.default_timer()
y_true = y_test
matrix = confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4,5,6,7,8,9])
nb_predictions = len(y_true)
print("\nKNN_optim")
print('Time (KNN_optim): ', stop - start) 
print("p =",p)
print("Nombre de prédictions : ",nb_predictions)
nb_bonnes_predictions = 0
for i in range(nb_predictions):
	if y_test[i]==y_pred[i]:
		nb_bonnes_predictions += 1
score = nb_bonnes_predictions / nb_predictions
print("Nombre de bonnes prédictions : ", nb_bonnes_predictions )
print("Nombre d'erreurs : ", nb_predictions-nb_bonnes_predictions )
print("Score : ",score)
print("Matrice de confusion :")
print(matrix)
"""










# Ici on regarde les scores en fonction de p
s_en_fonction_de_p = 0

perfore_X_test = 1 # si on veut perforer X_test
remplissage_blanc_images_X_test = 0 # si on veut remplir les images de X_test avec des pixels blancs
remplissage_aleatoire_images_X_test = 0 # si on veut remplir les images de X_test avec des pixels blancs

perfore_X_train = 1 # si on veut perforer X_train
reconstruction_images_X_train = 0 # si on veut reconstruire ou non les images de X_train
remplissage_blanc_images_X_train = 0 # si on veut remplir X_train avec des pixels blancs
remplissage_aleatoire_images_X_train = 0 # si on veut remplir X_train avec des pixels aléatoires
remplissage_moyenne_images_X_train = 1 # si on veut remplir X_train avec des pixels moyennés


if s_en_fonction_de_p==1:
	nombre_de_points = 100
	p_max = 1.0
	p_min = 0.01
	p_range = np.linspace(p_max,p_min,nombre_de_points)
	scores = []
	for p in p_range: # on fait une boucle sur plusieurs p
		# On perfore les données X_train avec des trous :
		X_train_filtered = []
		for chiffre in chiffres:
			X_train_filtered.append(np.array(X_train_filtered_original[chiffre]))

		# On perfore les données X_test avec des trous :
		X_test = np.array(X_test_original)
		if perfore_X_test==1:
			X_test = perforation_images(X=X_test,p=p)
			# Ici on remplis les images de X_test avec des pixels blancs
			if remplissage_blanc_images_X_test==1:
				X_test = remplissage_blanc_images(X=X_test)
			# Ici on remplis les images de X_test avec des pixels aléatoires
			if remplissage_aleatoire_images_X_test==1:
				X_test = remplissage_aleatoire_images(X=X_test)
		if perfore_X_train==1:
			X_train_filtered = perforation_images_filtered(X_filtered=X_train_filtered,p=p)
			# Ici on remplis les images de X_train avec des pixels blancs
			if remplissage_blanc_images_X_train==1:
				X_train_filtered = remplissage_blanc_images_filtered(X_filtered=X_train_filtered)
			# Ici on remplis les images de X_train avec des pixels aléatoires
			if remplissage_aleatoire_images_X_train==1:
				X_train_filtered = remplissage_aleatoire_images_filtered(X_filtered=X_train_filtered)
			# Ici on remplis les images de X_train avec des pixels moyennés
			if remplissage_moyenne_images_X_train==1:
				X_train_filtered = remplissage_moyenne_images_filtered(X_filtered=X_train_filtered)
			# Ici on reconstruit les images de X_train
			if reconstruction_images_X_train==1:
				start = timeit.default_timer()
				phases_reconstruction = 1
				for phase in range(phases_reconstruction): # on peut reconstruire plusieurs fois
					if phase>0: # si on reconstruit plusieurs fois, on perfore avant chaque reconstruction
						print("\nPhase de reconstruction : ",phase)
						X_train_filtered = perforation_images_filtered(X_filtered=X_train_filtered,p=p)
					X_train_filtered = reconstruction_images_filtered(X_train_filtered=X_train_filtered)
				stop = timeit.default_timer() 
				print('Temps pris pour reconstruire les images : ', stop - start) # si on fait ne manipule pas les images c'est 0.001=1/1000 seconde
			X_train = np.vstack(X_train_filtered)

		# On regarde les résultats
		y_pred = KNN(y_train=y_train,X_train=X_train,X_test=X_test)
		y_true = y_test
		matrix = confusion_matrix(y_true, y_pred, labels=chiffres)
		nb_predictions = len(y_true)
		nb_bonnes_predictions = np.count_nonzero(np.array(y_test)==np.array(y_pred))
		score = nb_bonnes_predictions / nb_predictions
		scores.append(score)
		#print("\np =",p)
		#print("Nombre de prédictions : ",nb_predictions)
		#print("Nombre de bonnes prédictions : ", nb_bonnes_predictions )
		#print("Nombre d'erreurs : ", nb_predictions-nb_bonnes_predictions )
		print("Score : ",score)
		print("p : ",p)
		#print("Matrice de confusion :")
		#print(matrix)

	x = p_range
	y = scores
	string = "p = np.array(["
	for i in range(len(x)):
		if i<len(x)-1:
			string+= "%.4f"%x[i] + ","
		if i==len(x)-1:
			string+= "%.4f"%x[i] + "])"
	print(string)
	string = "s = np.array(["
	for i in range(len(y)):
		if i<len(y)-1:
			string+= "%.4f"%y[i] + ","
		if i==len(y)-1:
			string+= "%.4f"%y[i] + "])"
	print(string)
	print("\n")
	#print("p_avec = np.concatenate((p_avec,p_avec_new))")
	#print("s_avec = np.concatenate((s_avec,s_avec_new))")





"""

Ici c'est X_train et X_test qui sont perforés
L'algo de reconstruction de X_train est ici activé
Une seule reconstruction est faite (i.e. phases=1)

p = 1.0
Nombre de prédictions :  450
Nombre de bonnes prédictions :  442
Nombre d'erreurs :  8
Score :  0.9822222222222222
Matrice de confusion :
[[39  0  0  0  0  0  0  0  0  0]
 [ 0 37  0  0  0  0  0  0  0  0]
 [ 0  0 43  0  0  0  0  0  0  0]
 [ 0  0  0 46  0  0  0  0  0  0]
 [ 0  0  0  0 50  0  0  0  0  0]
 [ 0  0  0  0  0 48  1  0  0  0]
 [ 0  0  0  0  0  0 52  0  0  0]
 [ 0  0  0  0  0  0  0 38  0  0]
 [ 0  3  0  0  0  0  0  0 41  1]
 [ 0  0  0  2  0  0  0  0  1 48]]

p = 0.9
Nombre de prédictions :  450
Nombre de bonnes prédictions :  442
Nombre d'erreurs :  8
Score :  0.9822222222222222
Matrice de confusion :
[[39  0  0  0  0  0  0  0  0  0]
 [ 0 37  0  0  0  0  0  0  0  0]
 [ 0  0 43  0  0  0  0  0  0  0]
 [ 0  0  0 45  0  1  0  0  0  0]
 [ 0  0  0  0 50  0  0  0  0  0]
 [ 0  0  0  0  0 48  1  0  0  0]
 [ 0  0  0  0  0  0 52  0  0  0]
 [ 0  0  0  0  0  0  0 38  0  0]
 [ 0  2  0  0  0  0  0  1 42  0]
 [ 0  0  0  2  0  0  0  0  1 48]]

p = 0.8
Nombre de prédictions :  450
Nombre de bonnes prédictions :  440
Nombre d'erreurs :  10
Score :  0.9777777777777777
Matrice de confusion :
[[39  0  0  0  0  0  0  0  0  0]
 [ 0 37  0  0  0  0  0  0  0  0]
 [ 0  0 43  0  0  0  0  0  0  0]
 [ 0  0  0 46  0  0  0  0  0  0]
 [ 0  0  0  0 49  0  0  1  0  0]
 [ 0  0  0  0  0 48  1  0  0  0]
 [ 0  0  0  0  0  0 52  0  0  0]
 [ 0  0  0  0  0  0  0 38  0  0]
 [ 0  3  0  0  0  0  0  1 41  0]
 [ 0  0  0  3  0  0  0  0  1 47]]

p = 0.7
Nombre de prédictions :  450
Nombre de bonnes prédictions :  437
Nombre d'erreurs :  13
Score :  0.9711111111111111
Matrice de confusion :
[[39  0  0  0  0  0  0  0  0  0]
 [ 0 36  0  0  0  0  1  0  0  0]
 [ 0  0 43  0  0  0  0  0  0  0]
 [ 0  0  0 46  0  0  0  0  0  0]
 [ 0  1  0  0 49  0  0  0  0  0]
 [ 0  0  0  0  1 48  0  0  0  0]
 [ 1  0  0  0  0  0 51  0  0  0]
 [ 0  0  0  0  1  0  0 37  0  0]
 [ 0  4  0  0  0  0  0  0 41  0]
 [ 0  0  0  2  0  1  0  0  1 47]]

p = 0.6
Nombre de prédictions :  450
Nombre de bonnes prédictions :  421
Nombre d'erreurs :  29
Score :  0.9355555555555556
Matrice de confusion :
[[39  0  0  0  0  0  0  0  0  0]
 [ 0 35  0  0  1  0  0  0  1  0]
 [ 0  1 41  0  0  0  0  0  1  0]
 [ 0  0  0 39  0  0  1  0  2  4]
 [ 0  0  0  0 49  0  0  1  0  0]
 [ 0  0  0  0  0 48  1  0  0  0]
 [ 0  0  0  0  0  0 51  0  0  1]
 [ 0  0  0  0  0  0  0 37  0  1]
 [ 1  3  0  1  0  0  0  0 36  4]
 [ 1  0  0  3  0  0  0  0  1 46]]

p = 0.5
Nombre de prédictions :  450
Nombre de bonnes prédictions :  409
Nombre d'erreurs :  41
Score :  0.9088888888888889
Matrice de confusion :
[[39  0  0  0  0  0  0  0  0  0]
 [ 0 35  1  0  1  0  0  0  0  0]
 [ 0  0 38  3  1  0  0  0  1  0]
 [ 0  1  1 41  0  1  1  0  0  1]
 [ 0  1  0  0 49  0  0  0  0  0]
 [ 0  0  0  1  1 46  0  0  1  0]
 [ 1  1  0  0  0  1 48  0  1  0]
 [ 0  0  0  0  0  0  0 38  0  0]
 [ 0  2  0  4  0  0  0  1 34  4]
 [ 1  0  0  3  0  2  0  0  4 41]]

p = 0.4
Nombre de prédictions :  450
Nombre de bonnes prédictions :  366
Nombre d'erreurs :  84
Score :  0.8133333333333334
Matrice de confusion :
[[37  0  0  0  0  1  0  1  0  0]
 [ 0 31  0  0  2  0  1  1  1  1]
 [ 0  2 33  2  1  0  1  2  2  0]
 [ 0  0  1 34  0  2  0  1  1  7]
 [ 1  0  0  0 45  0  1  2  1  0]
 [ 1  0  1  2  1 42  0  1  1  0]
 [ 0  3  1  0  3  0 45  0  0  0]
 [ 0  1  1  0  0  1  0 34  1  0]
 [ 0  4  1  3  0  4  2  1 29  1]
 [ 0  1  0  5  0  1  1  2  5 36]]

p = 0.3
Nombre de prédictions :  450
Nombre de bonnes prédictions :  335
Nombre d'erreurs :  115
Score :  0.7444444444444445
Matrice de confusion :
[[34  0  0  1  1  0  2  0  0  1]
 [ 0 30  1  0  0  1  0  2  3  0]
 [ 1  1 31  3  0  1  2  3  1  0]
 [ 1  0  1 31  0  8  0  0  0  5]
 [ 3  1  0  0 40  0  1  3  2  0]
 [ 2  2  0  0  2 34  0  2  2  5]
 [ 2  2  0  0  2  0 45  0  1  0]
 [ 0  0  0  1  0  1  0 34  1  1]
 [ 1  6  1  2  1  1  0  2 27  4]
 [ 1  1  1  7  1  4  1  1  5 29]]

p = 0.2
Nombre de prédictions :  450
Nombre de bonnes prédictions :  263
Nombre d'erreurs :  187
Score :  0.5844444444444444
Matrice de confusion :
[[32  1  0  2  1  0  1  0  1  1]
 [ 0 14  2  1  3  3  2  1  7  4]
 [ 0  2 22  6  1  3  2  1  5  1]
 [ 0  1  9 23  0  5  1  0  1  6]
 [ 3  3  0  1 25  3  7  5  2  1]
 [ 0  3  1  4  2 26  2  3  2  6]
 [ 5  1  3  0  2  0 37  1  2  1]
 [ 1  0  3  0  1  1  1 29  1  1]
 [ 3  3  3  2  1  4  0  5 22  2]
 [ 2  3  1  5  0  3  1  2  1 33]]

p = 0.1
Nombre de prédictions :  450
Nombre de bonnes prédictions :  155
Nombre d'erreurs :  295
Score :  0.34444444444444444
Matrice de confusion :
[[23  0  1  4  1  2  2  0  1  5]
 [ 3 19  1  0  3  3  5  2  0  1]
 [ 5  4 15  4  0  2  6  2  2  3]
 [ 1  1 16 15  0  2  1  3  3  4]
 [ 6  7  1  2 22  3  0  5  3  1]
 [ 4  8  3  9  4 10  3  4  1  3]
 [ 4 11  2  5  5  2 15  3  3  2]
 [ 1  5  2  3  4  3  1 13  3  3]
 [ 5  6  4  6  2  2  4  2 10  4]
 [ 7  3  2  6  1  7  4  5  3 13]]


"""

############################################

"""

Ici c'est autant X_train et X_test qui sont perforés
Et il n'y a pas de reconstruction d'images

Avec p=1.0 (i.e. aucun mauvais pixel) on a 98.22% as usual avec KNN :

p=1.0
Nombre de prédictions :  450
Nombre de bonnes prédictions :  442
Nombre d'erreurs :  8
Score :  0.9822222222222222
Matrice de confusion :
[[39  0  0  0  0  0  0  0  0  0]
 [ 0 37  0  0  0  0  0  0  0  0]
 [ 0  0 43  0  0  0  0  0  0  0]
 [ 0  0  0 46  0  0  0  0  0  0]
 [ 0  0  0  0 50  0  0  0  0  0]
 [ 0  0  0  0  0 48  1  0  0  0]
 [ 0  0  0  0  0  0 52  0  0  0]
 [ 0  0  0  0  0  0  0 38  0  0]
 [ 0  3  0  0  0  0  0  0 41  1]
 [ 0  0  0  2  0  0  0  0  1 48]]


p = 0.9
Nombre de prédictions :  450
Nombre de bonnes prédictions :  440
Nombre d'erreurs :  10
Score :  0.9777777777777777
Matrice de confusion :
[[39  0  0  0  0  0  0  0  0  0]
 [ 0 37  0  0  0  0  0  0  0  0]
 [ 0  0 43  0  0  0  0  0  0  0]
 [ 0  0  0 46  0  0  0  0  0  0]
 [ 0  0  0  0 50  0  0  0  0  0]
 [ 0  0  0  0  0 49  0  0  0  0]
 [ 0  0  0  0  0  0 52  0  0  0]
 [ 0  0  0  0  0  0  0 38  0  0]
 [ 0  4  0  0  0  0  0  0 40  1]
 [ 0  0  0  4  0  0  0  0  1 46]]

p = 0.8
Nombre de prédictions :  450
Nombre de bonnes prédictions :  434
Nombre d'erreurs :  16
Score :  0.9644444444444444
Matrice de confusion :
[[39  0  0  0  0  0  0  0  0  0]
 [ 0 37  0  0  0  0  0  0  0  0]
 [ 0  0 42  0  0  0  0  0  1  0]
 [ 0  0  0 46  0  0  0  0  0  0]
 [ 0  2  0  0 48  0  0  0  0  0]
 [ 0  0  0  0  1 47  1  0  0  0]
 [ 1  0  0  0  0  0 51  0  0  0]
 [ 0  0  0  0  0  0  0 38  0  0]
 [ 0  4  0  0  0  0  0  0 38  3]
 [ 0  0  0  2  0  0  0  0  1 48]]

p = 0.7
Nombre de prédictions :  450
Nombre de bonnes prédictions :  405
Nombre d'erreurs :  45
Score :  0.9
Matrice de confusion :
[[39  0  0  0  0  0  0  0  0  0]
 [ 0 37  0  0  0  0  0  0  0  0]
 [ 0  2 40  0  0  0  0  0  1  0]
 [ 0  0  1 42  1  0  0  0  0  2]
 [ 0  0  0  0 50  0  0  0  0  0]
 [ 0  0  0  0  2 45  1  1  0  0]
 [ 1  1  0  0  1  1 48  0  0  0]
 [ 0  0  0  0  0  0  0 35  2  1]
 [ 1  6  1  2  1  0  1  0 30  3]
 [ 0  0  0  6  0  4  0  0  2 39]]

p = 0.6
Nombre de prédictions :  450
Nombre de bonnes prédictions :  367
Nombre d'erreurs :  83
Score :  0.8155555555555556
Matrice de confusion :
[[36  0  0  0  2  0  0  0  1  0]
 [ 0 35  0  0  0  0  1  1  0  0]
 [ 0  0 38  2  0  0  1  2  0  0]
 [ 0  3  0 37  0  0  2  1  2  1]
 [ 1  1  0  0 44  0  0  4  0  0]
 [ 0  0  2  2  3 38  2  0  0  2]
 [ 1  1  0  0  6  0 44  0  0  0]
 [ 0  1  1  0  0  0  0 36  0  0]
 [ 3  5  3  2  1  1  2  1 26  1]
 [ 0  2  3  5  0  2  1  0  5 33]]

p = 0.5
Nombre de prédictions :  450
Nombre de bonnes prédictions :  295
Nombre d'erreurs :  155
Score :  0.6555555555555556
Matrice de confusion :
[[32  0  0  0  1  2  3  0  0  1]
 [ 0 26  3  1  4  0  1  0  1  1]
 [ 0  1 27  3  2  2  3  2  3  0]
 [ 0  3  3 30  0  5  0  3  1  1]
 [ 1  2  0  0 33  1  3  8  2  0]
 [ 2  3  0  2  1 32  4  1  1  3]
 [ 2  5  1  0  1  2 40  0  1  0]
 [ 0  4  1  0  1  1  1 29  0  1]
 [ 1  6  0  5  2  3  0  6 18  4]
 [ 1  2  1  6  0  6  2  1  4 28]]

p = 0.4
Nombre de prédictions :  450
Nombre de bonnes prédictions :  123
Nombre d'erreurs :  327
Score :  0.2733333333333333
Matrice de confusion :
[[14  1  1  0  1  1  9  5  3  4]
 [ 1 15  3  3  4  2  3  1  2  3]
 [ 4  7  9  4  0  9  3  5  2  0]
 [ 4  6  4 11  1  3  3  4  2  8]
 [ 5 12  1  2 13  2  1  6  2  6]
 [ 4  5  2  4  1 14  4  4  5  6]
 [ 2  4  1  2  2 10 17  3  5  6]
 [ 1 10  1  4  2  1  2 13  0  4]
 [ 3  9  2  4  5  6  0  2  7  7]
 [ 7 11  2  4  1  9  1  5  1 10]]

p = 0.3
Nombre de prédictions :  450
Nombre de bonnes prédictions :  77
Nombre d'erreurs :  373
Score :  0.1711111111111111
Matrice de confusion :
[[10  2  5  2  3  3  6  5  3  0]
 [ 2  8  4  4  6  0  8  4  1  0]
 [ 5  7  7  5  2  2  9  3  2  1]
 [ 5  3  3  7  4  3 11  5  5  0]
 [ 5  5  1  2 11  4 10  4  7  1]
 [ 4  3  6  9  5  4 11  2  5  0]
 [ 3  8  7  3  2  2 20  2  5  0]
 [ 3  9  1  2  5  2  7  7  0  2]
 [ 1  9  8  7  3  2  9  2  2  2]
 [ 3  3  4  8  7  5 11  5  4  1]]

p = 0.2
Nombre de prédictions :  450
Nombre de bonnes prédictions :  69
Nombre d'erreurs :  381
Score :  0.15333333333333332
Matrice de confusion :
[[12  3  3  8  6  1  3  2  0  1]
 [ 4  3  3  7 12  2  5  1  0  0]
 [ 6  0  6 13 11  3  1  1  1  1]
 [ 5  1  5 13 14  2  4  0  1  1]
 [ 9  5  2  8 20  1  3  1  1  0]
 [ 3  3  8  4 17  6  2  3  1  2]
 [ 8  3  6 14 14  0  3  2  0  2]
 [ 6  2  4  3 14  2  1  3  1  2]
 [ 6  2  5  7 16  3  3  2  0  1]
 [ 3  7  7  5  9  5  4  7  1  3]]

p = 0.1
Nombre de prédictions :  450
Nombre de bonnes prédictions :  59
Nombre d'erreurs :  391
Score :  0.13111111111111112
Matrice de confusion :
[[ 8  3  2 12  9  2  2  1  0  0]
 [ 8  6  5  6  6  3  1  1  0  1]
 [ 3  6  6 14  7  5  0  1  1  0]
 [ 1  6  7 11  9  5  7  0  0  0]
 [ 4  5  0 13 15  4  3  2  0  4]
 [ 8  5  6 10  7  8  1  2  2  0]
 [ 4  7  7  9 16  6  1  1  1  0]
 [ 4  2  5  4 14  3  2  2  1  1]
 [ 3  9  4  8  9  8  3  1  0  0]
 [ 5  8  4 12 10  7  2  1  0  2]]

p = 0.0
Nombre de prédictions :  450
Nombre de bonnes prédictions :  50
Nombre d'erreurs :  400
Score :  0.1111111111111111
Matrice de confusion :
[[ 0  0  0  0 39  0  0  0  0  0]
 [ 0  0  0  0 37  0  0  0  0  0]
 [ 0  0  0  0 43  0  0  0  0  0]
 [ 0  0  0  0 46  0  0  0  0  0]
 [ 0  0  0  0 50  0  0  0  0  0]
 [ 0  0  0  0 49  0  0  0  0  0]
 [ 0  0  0  0 52  0  0  0  0  0]
 [ 0  0  0  0 38  0  0  0  0  0]
 [ 0  0  0  0 45  0  0  0  0  0]
 [ 0  0  0  0 51  0  0  0  0  0]]



"""





####################################################

# Ici on affiche diverses images






"""
# Pour afficher les digits 0 à 9 sur trois lignes (donc 27 chiffres sur 3 lignes)
nb_horiz = 10
nb_vert = 3
# Visualisation des chiffres
fig = plt.figure(figsize=(6,6))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
for i in range(nb_horiz*nb_vert):
	ax = fig.add_subplot(nb_vert,nb_horiz,i+1,xticks=[],yticks=[])
	# On transforme en array d'images
	w,h=8,8 # images 8x8
	image = np.zeros((w, h))
	for j in range(8):
		image[j] = X_normed[i][8*j:8*j+8]
	img = Image.fromarray(image)
	ax.imshow(img, cmap=plt.cm.binary, interpolation='nearest')
	#ax.imshow(X[i], cmap=plt.cm.binary, interpolation='nearest')
	ax.text(0,7,str(digits.target[i]))
plt.show()
"""

"""
# Pour afficher une ligne de digits avec données manquantes (trous = gris)
nb_horiz = 10
nb_vert = 1
# Visualisation des chiffres
fig = plt.figure(figsize=(6,6))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
for i in range(nb_horiz*nb_vert):
	ax = fig.add_subplot(nb_vert,nb_horiz,i+1,xticks=[],yticks=[])
	# On transforme en array d'images
	w,h=8,8 # images 8x8
	image = np.zeros((h,w))
	picture = X_train_filtered[i][0]
	for j in range(8):
		image[j] = picture[8*j:8*j+8] # À corriger, je veux pixel=-1 implique rouge
	img = Image.fromarray(image)
	ax.imshow(img, cmap=plt.cm.binary, interpolation='nearest')
	#ax.imshow(X[i], cmap=plt.cm.binary, interpolation='nearest')
	ax.text(0,7,str(digits.target[i]))
plt.show()
"""

"""
# Pour afficher une ligne de digits avec données manquantes (trous = rouge)
# Ici c'est basé sur les données déjà perforées
nb_horiz = 10
nb_vert = 1
# Visualisation des chiffres
fig = plt.figure(figsize=(6,6))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
for i in range(nb_horiz*nb_vert):
	ax = fig.add_subplot(nb_vert,nb_horiz,i+1,xticks=[],yticks=[])
	# On transforme en array d'images
	h,w=8,8 # images 8x8
	#image = Image.new(mode='RGB', size=(w,h), color=0)
	image = np.zeros((h,w,3)) # ici on prend une image couleur
	picture = X_train_filtered[i][0]
	for vert in range(h):
		for horiz in range(w):
			pixel = picture[w*vert + horiz]
			if pixel==-1:
				image[vert][horiz] = [255,0,0]
			else:
				image[vert][horiz] = [int(255.0*(1-pixel)),int(255.0*(1-pixel)),int(255.0*(1-pixel))]
	ax.imshow(image.astype(np.uint8),vmin=0, vmax=255)
	#print(image)
	ax.text(0,7,str(digits.target[i]))
plt.show()
# Pour faire ça rapidement avec plusieurs valeurs de p j'ai juste à perforer les données live pour l'affichage.
"""

"""
# Pour afficher les digits 0 à 9 sur trois lignes (donc 27 chiffres sur 3 lignes)
nb_horiz = 10
nb_vert = 3
w,h=8,8 # images 8x8
# Visualisation des chiffres
fig = plt.figure(figsize=(6,6))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
for i in range(nb_horiz*nb_vert):
	ax = fig.add_subplot(nb_vert,nb_horiz,i+1,xticks=[],yticks=[])
	# On transforme en array d'images
	image = np.zeros((h,w,3)) # ici on prend une image couleur
	for vert in range(h):
		for horiz in range(w):
			pixel = X_normed[i][w*vert+horiz]
			good_or_not = np.random.choice(population, p=weights)
			if good_or_not==0: # si le pixel n'est pas bon on y met -1
				image[vert][horiz] = [255,0,0]
			else:
				image[vert][horiz] = [int(255.0*(1-pixel)),int(255.0*(1-pixel)),int(255.0*(1-pixel))]
	ax.imshow(image.astype(np.uint8),vmin=0, vmax=255)
	ax.text(0,7,str(digits.target[i]))
plt.show()
"""


# ok, je suis rendu à mettre les pixels en rouge graduellement
# ici c'est pas sigma qui varie mais bien p = [1,0.9,0.8,0.7,...,0]


"""
# Autre manière d'afficher ça en grille, plus clair en termes des indices (i,j) = (ligne,colonne)
proportions = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
nb_horiz = 10
nb_vert = len(proportions)
w,h=8,8 # images 8x8
size = w*h # 64 pixels
fig = plt.figure(figsize=(6,6))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
for i in range(nb_vert): # (i,j) = (ligne,colonne)
	proportion = proportions[i]
	for j in range(nb_horiz):
		image = np.zeros((h,w,3)) # ici on prend une image couleur
		for vert in range(h):
			for horiz in range(w):
				pixel = X_normed[j][w*vert+horiz]
				good_or_not = np.random.choice([0,1], p=[1-proportion,proportion])
				if good_or_not==0: # si le pixel n'est pas bon on y met -1
					image[vert][horiz] = [255,0,0]
				else:
					image[vert][horiz] = [int(255.0*(1-pixel)),int(255.0*(1-pixel)),int(255.0*(1-pixel))]

		# On crée la case et on la met dans la grille
		ax = fig.add_subplot(nb_vert,nb_horiz,1+10*i+j,xticks=[],yticks=[])
		ax.imshow(image.astype(np.uint8),vmin=0, vmax=255)
		ax.text(0,7,str(j))
plt.show()
"""

"""
# Autre manière en grille, avec colonne pour la proportion
proportions = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
# proportion est la proportion de bons pixels
nb_horiz = 10
nb_vert = len(proportions)
w,h=8,8 # images 8x8
size = w*h # 64 pixels
fig = plt.figure(figsize=(6,6))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
for i in range(nb_vert): # (i,j) = (ligne,colonne)
	proportion = proportions[i]
	for j in range(nb_horiz+1):
		image = np.zeros((h,w,3)) # ici on prend une image couleur
		image.fill(255) # on met tout en blanc
		if j==0: # pour afficher la proportion p
			ax = fig.add_subplot(nb_vert,nb_horiz+1,1+10*i+i,xticks=[],yticks=[])
			ax.imshow(image.astype(np.uint8),vmin=0, vmax=255)
			ax.text(0,4,"p ="+ "%.1f"%proportion)

		if j!=0:
			for vert in range(h):
				for horiz in range(w):
					pixel = X_normed[j-1][w*vert+horiz]
					good_or_not = np.random.choice([0,1], p=[1-proportion,proportion])
					if good_or_not==0: # si le pixel n'est pas bon on y met -1
						image[vert][horiz] = [255,0,0]
					else:
						image[vert][horiz] = [int(255.0*(1-pixel)),int(255.0*(1-pixel)),int(255.0*(1-pixel))]
			# On crée la case et on la met dans la grille
			ax = fig.add_subplot(nb_vert,nb_horiz+1,1+10*i+j+i,xticks=[],yticks=[])
			ax.imshow(image.astype(np.uint8),vmin=0, vmax=255)
			ax.text(0,7,str(j))
plt.show()
"""


"""
# Ici on affiche deux images et on fait un & pour joindre les pixels manquants
# On s'intéresse aux paires de pixels corrects.
nb_horiz = 5
nb_vert = 1
# Visualisation des chiffres
fig = plt.figure(figsize=(6,6))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
picture1 = []
picture2 = []
picture3 = []
pictures = [picture1,picture2,picture3]
for i in range(64):
	picture1.append(np.random.choice(population, p=weights))
	picture2.append(np.random.choice(population, p=weights))
	picture3.append(picture1[i]*picture2[i])
for i in range(nb_horiz*nb_vert):
	ax = fig.add_subplot(nb_vert,nb_horiz,i+1,xticks=[],yticks=[])
	# On transforme en array d'images
	h,w=8,8 # images 8x8
	#image = Image.new(mode='RGB', size=(w,h), color=0)
	image = np.zeros((h,w,3)) # ici on prend une image couleur
	image.fill(255)
	if i==1:
		ax.text(3,4,"&",fontsize=40)
	if i==3:
		ax.text(3,4,"=",fontsize=40)
	if i%2==0: # si c'est paire on met une image verte et rouge
		picture = pictures[int(i/2)]
		for vert in range(h):
			for horiz in range(w):
				pixel = picture[w*vert+horiz]
				if pixel==0: # bad pixel
					image[vert][horiz] = [255,0,0] # red
				if pixel==1: # good pixel
					image[vert][horiz] = [0,255,0] # green
	ax.imshow(image.astype(np.uint8),vmin=0, vmax=255)
plt.show()
# Pour faire ça rapidement avec plusieurs valeurs de p j'ai juste à perforer les données live pour l'affichage.
"""




"""
# On crée des images :
X_train_filtered_perfore = perforation_images_filtered(X_filtered=X_train_filtered_original,p=p) # on perfore X_train
X_train_filtered_blanc = remplissage_blanc_images_filtered(X_filtered=X_train_filtered_perfore) # remplissage blanc
X_train_filtered_aleatoire = remplissage_aleatoire_images_filtered(X_filtered=X_train_filtered_perfore) # remplissage aléatoire
X_train_filtered_reconstruit = reconstruction_images_filtered(X_train_filtered=X_train_filtered_perfore) # reconstruction
X_train_filtered_moyenne = remplissage_moyenne_images_filtered(X_filtered=X_train_filtered_perfore) # remplissage moyenné
# On se retrouve alors avec 5 colonnes à afficher :
X_train_filtered_perfore # les images perforées (plein de rouge)
X_train_filtered_blanc # les images perforées remplies de blanc
X_train_filtered_aleatoire # les images perforées remplies aléatoirement
X_train_filtered_reconstruit # les images perforées reconstruites
X_train_filtered_moyenne # les images perforées avec remplissage moyenné
# Ça avec une colonne p=1.0, p=0.9, ..., p=0.1 ça fait 6 colonnes en tout.
"""
"""
nombre_de_points = 10
p_max = 1.0
p_min = 0.1
p_range = np.linspace(p_max,p_min,nombre_de_points) # [1.  0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1]
chiffre = 6 # on va faire ça pour le chiffre 6
"""


# Ici on affiche un chiffre pour différents p et différentes méthodes de remplissage et de reconstruction
# On s'intéresse aux paires de pixels corrects.
nb_horiz = 5
nb_vert = 10
w,h=8,8 # images 8x8
size = w*h # 64 pixels
# Visualisation des chiffres
fig = plt.figure(figsize=(6,6))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
chiffre = 6
fontsize = 7
p_range = np.linspace(1.0,0.1,10) # [1.  0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1]
for j in range(nb_horiz+1):
	if j>0:
		image = np.zeros((h,w,3)) # ici on prend une image couleur
		image.fill(255) # on met tout en blanc
		ax = fig.add_subplot(nb_vert+1,nb_horiz+1,1+j,xticks=[],yticks=[])
		ax.imshow(image.astype(np.uint8),vmin=0, vmax=255)
		if j==1:ax.text(0,4,"Perforé",fontsize=fontsize)
		if j==2:ax.text(0,4,"Blanc",fontsize=fontsize)
		if j==3:ax.text(0,4,"Aléatoire",fontsize=fontsize)
		if j==4:ax.text(0,4,"Reconstruit",fontsize=fontsize)
		if j==5:ax.text(0,4,"Moyenne",fontsize=fontsize)

for i in range(10):
	p = p_range[i]
	X_train_chiffre             = np.array(X_train_filtered_original[chiffre])
	X_train_chiffre_perfore     = perforation_images(X=X_train_chiffre,p=p)
	X_train_chiffre_blanc       = remplissage_blanc_images(X=np.array(X_train_chiffre_perfore))
	X_train_chiffre_aleatoire   = remplissage_aleatoire_images(X=np.array(X_train_chiffre_perfore))
	X_train_chiffre_reconstruit = reconstruction_images_chiffre(X_train_chiffre=np.array(X_train_chiffre_perfore))
	#X_train_chiffre_reconstruit = remplissage_aleatoire_images(X=np.array(X_train_chiffre_perfore)) # temporaire pour accélérer calcul
	X_train_chiffre_moyenne     = remplissage_moyenne_images_chiffre(X=np.array(X_train_chiffre_perfore))
	image_perfore     = np.array(X_train_chiffre_perfore[0])
	image_blanc       = np.array(X_train_chiffre_blanc[0])
	image_aleatoire   = np.array(X_train_chiffre_aleatoire[0])
	image_reconstruit = np.array(X_train_chiffre_reconstruit[0])
	image_moyenne     = np.array(X_train_chiffre_moyenne[0])
	#print(image_moyenne)
	images = [image_perfore,image_blanc,image_aleatoire,image_reconstruit,image_moyenne]
	for j in range(nb_horiz+1):
		image = np.zeros((h,w,3)) # ici on prend une image couleur
		image.fill(255) # on met tout en blanc
		if j==0: # pour afficher la proportion p
			ax = fig.add_subplot(nb_vert+1,nb_horiz+1,1+5*i+i+6,xticks=[],yticks=[])
			ax.imshow(image.astype(np.uint8),vmin=0, vmax=255)
			ax.text(0,4,"p ="+ "%.1f"%p)
		if j!=0:
			for vert in range(h):
				for horiz in range(w):
					pixel = images[j-1][w*vert+horiz]
					if pixel==-1:
						image[vert][horiz] = [255,0,0]
					else:
						image[vert][horiz] = [int(255.0*(1-pixel)),int(255.0*(1-pixel)),int(255.0*(1-pixel))]
			# On crée la case et on la met dans la grille
			ax = fig.add_subplot(nb_vert+1,nb_horiz+1,1+5*i+j+i+6,xticks=[],yticks=[])
			ax.imshow(image.astype(np.uint8),vmin=0, vmax=255)
			#ax.text(0,7,str(j))
plt.show()














