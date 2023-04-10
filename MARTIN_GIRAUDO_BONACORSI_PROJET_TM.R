#MARTIN Erwan
#GIRAUDO Victorien
#BONACORSI Joshua
#ING 3 DS
#TEXT MINING PROJECT NOVEMBER 2022
#Importing of the necessary packages
library("tm")
library("readtext")
library(wordcloud)
library(stopwords)
library(gmp)
library(Rmpfr)
library(arules)
library("stringr") 
library("geometry")

#Work Directory
setwd("h:/Desktop/ING3/S1/Text mining/MARTIN_GIRAUDO_BONACORSI_PROJET_TM/Reuters_Dataset")

#PARTIE I
#Cleaning the data: write the R code and show an example in which you apply it to a subset of the data.

#CLEANING FUNCTION TO CLEAN A CORPUS

cleaning = function(corpus){
  #Remove suppercase 
  corpus = tm_map(corpus, content_transformer(tolower))
  #Remove punctuation
  corpus = tm_map(corpus,removePunctuation)
  #Remove stopwords
  stw = stopwords("english")
  corpus = tm_map(corpus,removeWords,stopwords("english"))
  corpus = tm_map(corpus, removeNumbers)
  return(corpus)
  
}

#Example of application

example = readtext(file = "test/bop/0010454")
corpus_example = VCorpus(VectorSource(example$text))

#Document before cleaning
content(corpus_example[[1]])

#Cleaning
cleaned_example = cleaning(corpus_example)

#Document after cleaning
content(cleaned_example[[1]])

#PARTIE II : STATISTICS

#II.A) Show the high dimensionality and sparsity of the data.
data = readtext(file = "training/acq/*")
corpus_data= VCorpus(VectorSource(data$text))
dtm = DocumentTermMatrix(corpus_data)

dtm
dim(dtm)
M = as.matrix(dtm)
M[M>0] = 1 
nb_coeff_non_nul = sum(M) # 118 252 non-zero coefficients in the word matrix of the corpus
nb_coeff_non_nul
nb_coeff = length(M) #32,392,800 coeff in the word matrix ==> proves the high dimensionality of the data
nb_coeff
sparsity = (1 -nb_coeff_non_nul/nb_coeff)*100
sparsity
#We have a parsimony of 99.6% which shows the sparsity of the data ==> 99.6% of the coefficients of the word matrix are zero

#II.B) Select some topics in both training and test sets and represent them by wordclouds. Comment the result.

#Function to display the wordcloud of a corpus

wc = function(data){
  crp =  VCorpus(VectorSource(data$text))
  clean_crp = cleaning(crp)
  dtm = DocumentTermMatrix(clean_crp)
  m = as.matrix(dtm)
  m[m>0] = 1
  frequency = colSums(m)
  frequency = sort(frequency, decreasing = TRUE)
  words = names(frequency)
  wordcloud(words[1:50],frequency[1:50],scale=c(2,1))
}

#Exemple wordcloud 1 : Topic Trade

#Importing documents from the trade topic
data_trade_training = readtext(file = "training/trade/*")
data_trade_test = readtext(file="test/trade/*")

#Training Wordcloud
wc(data_trade_training)

#Test Wordcloud
wc(data_trade_test)

#On remarque que les mots "trade","said","japan","japanese","washington"
#Donc on a une certaine similarit� entre le wordcloud du training set et du testing set

#REMARQUE: Les images sont enregistr�es dans le dossier pour mieux visualiser

#Exemple wordcloud 2 : Topic ship

data_ship_training = readtext(file = "training/ship/*")
data_ship_test = readtext(file = "test/ship/*")

#Wordcloud du training
wc(data_ship_training)

#Wordcloud du test
wc(data_ship_test)

#We notice that the words "trade", "said", "japan", "japanese", "washington"
#So we have some similarity between the wordcloud of the training set and the testing set

#NB: The images are saved in the folder for better visualization
#We notice again the recurrence of words in the 2 sets: vessels,said,spokesman,ship,foreign...

#==> We can then say that the occurrence of some words can allow to determinate the class of a document
#II.C) Using a subset of your data, represent each document as a vector of TF-IDF values and 
#use these vectors to measure similarity between pairs of documents.

#Selected topics : trade, acq, ship, silver, tea

#Topics importation ( in addition to those already imported )
data_acq_training = readtext(file = "training/acq/*")
data_silver_training = readtext(file = "training/silver/*")
data_tea_training = readtext(file = "training/tea/*")

#Creation and Cleaning of the corpus

data_several_topics = c(data_trade_training$text,data_acq_training$text,data_ship_training$text,data_silver_training$text,data_tea_training$text)

#Corpus organization
# Documents 1 to 369 : Topic "Trade"
# Documents 370 to 2019 : Topic "acq"
# Documents 2020 to 2216 : Topic "ship"
# Documents 2217 to 2237 : Topic "silver"
# Documents 2238 to 2246 : Topic "tea"
crp =  VCorpus(VectorSource(data_several_topics))
clean_crp = cleaning(crp)

#Creation of the matrix TF-IDF

#Step 1 : TF matrix creation

dtm = DocumentTermMatrix(clean_crp,control = list(wordLengths=c(1,Inf))) 
#REMARQUE : L'argument wordLengths sert � consid�rer les mots inf�rieurs � 3 lettres pour �viter certains probl�mes dans la partie IV du projet

TF = t(as.matrix(dtm))
#View(TF) #La colonne j correspond au vecteur TF du document j sur l'ensemble des mots du corpus

#�tape 2 : Cr�ation de la matrice IDF

nb_doc = ncol(TF) #On r�cup�re le nombre de documents du corpus
#Cr�ation de la matrice binaire telle que le coefficient vaut 0 si le mot n'apparait pas dans le document et 1 sinon
presence_doc = as.matrix(dtm) # Il s'agit de la matrice de Term Frequency avant sa transposition
presence_doc[presence_doc>0]=1
#Somme sur les colonnes pour avoir le nombre de documents qui contiennent le mot en question
frequency = colSums(presence_doc) 
#IDF
idf = log(nb_doc/frequency,2) # idf associ� au terme w du corpus

#Matrice TF_IDF
tf_idf = idf*TF
#Conversion du tf_idf en dataframe pour simplifier son utilisation
tf_idf = as.data.frame(tf_idf)
#La colonne j correspond au vecteur TF-IDF du document j.
#View(tf_idf)

#Similarit� entre 2 documents 

#Fonction qui, en connaissant les vecteurs tf_idf de la paire de documents, va calculer la similarit� en donnant une valeur entre 0 et 1
#Sachant que 0 signifie que les deux documents sont tr�s peu similaires et 1 que les documents sont tr�s similaires

cos_similarity = function(v1,v2){
  norm_d1 = sqrt(dot(v1,v1))
  norm_d2 = sqrt(dot(v2,v2))
  ps = dot(v1,v2)
  
  return((ps)/(norm_d1*norm_d2))
}
#Conversion de la matrice tf_idf en objet matrice pour pouvoir faire des produits scalaires et autre calcul sur les colonnes
tf_idf_m = as.matrix(tf_idf)

#Exemple 1 : Similarit� d'un document avec lui-m�me

#R�cup�ration du vecteur tf_idf du document (ici le document num�ro 1 du corpus)
v1 = tf_idf_m[,1]
similarity_example1 = cos_similarity(v1,v1)
similarity_example1 #On trouve 1 

#Exemple 2 : Similarit� d'un document avec un document appartenant � la m�me classe

#R�cup�ration des vecteurs tf_idf des documents � comparer
#Ici on va comparer le document 1 au document 2 ( qui appartienne � la m�me classe "trade" d'apr�s la construction du corpus ( les 369 premiers sont de la classe trade )
v2 = tf_idf_m[,2]
similarity_example2 = cos_similarity(v1,v2)
similarity_example2 #On trouve 0.11

#Exemple 3 : Similarit� d'un document avec un docuement appartenant � une classe diff�rente
#R�cup�ration du vecteur tf_idf du document (ici le document num�ro 2000 du corpus donc un document de la classe acq)
v3 = tf_idf_m[,2000]
similarity_example3 = cos_similarity(v1,v3)
similarity_example3 #On trouve 0.0008==> valeur encore plus faible que si les documents appartenaient � la meme classe

#PARTIE III Classification : Create and test a Bayesian classifier for your data.

#III.A) You first need to represent each document as a bag of words.

#Fonction permettant de convertir un document en un sac de mot
toBagOfWords = function(document){
    lw = words(document)
    bag_of_words = list()
    for(word in lw){
      if(word %in% names(bag_of_words)){
        bag_of_words[[word]] =  bag_of_words[[word]]+1
      }
      else{
        bag_of_words[[word]] = 1
      }
      
    }
    return(bag_of_words)
  
}
#Exemple d'application sur le premier document du corpus 
d1 = clean_crp[[1]]
content(d1)
bw_d1 = toBagOfWords(d1)
bw_d1

#III.B) Classification will be applied to (a) subset(s) of the data containing less than 10 topics.

#On va faire de la classification sur les 5 topics de la partie II 

#Classes : trade, acq, ship, silver, tea

#Rappel de l'organisation du corpus 
# Documents 1 � 369 : Topic "Trade"
# Documents 370 � 2019 : Topic "acq"
# Documents 2020 � 2216 : Topic "ship"
# Documents 2217 � 2237 : Topic "silver"
# Documents 2238 � 2246 : Topic "tea"

#Pour cr�er le classifier, on a besoin de la probabilit� des classes, la probabilit� du document sachant la classe et la probabilit� du document

#PROBABILIT� DES CLASSES 

nb_tot_documents= length(data_several_topics) 
nb_tot_documents#2246 documents au total dans le training set

#Probabilit� de trade
nb_documents_classe_trade = length(data_trade_training$text) #369 documents de la classe trade
nb_documents_classe_trade
proba_trade  = nb_documents_classe_trade/nb_tot_documents

#Probabilit� de acq
nb_documents_classe_acq = length(data_acq_training$text)
nb_documents_classe_acq #1650 documents de classe acq
proba_acq  = nb_documents_classe_acq/nb_tot_documents

#Probabilit� de ship
nb_documents_classe_ship = length(data_ship_training$text)
nb_documents_classe_ship #197 documents de classe ship
proba_ship  = nb_documents_classe_ship/nb_tot_documents

#Probabilit� de silver
nb_documents_classe_silver = length(data_silver_training$text)
nb_documents_classe_silver #21 documents de classe ship
proba_silver  = nb_documents_classe_silver/nb_tot_documents

#Probabilit� de tea
nb_documents_classe_tea = length(data_tea_training$text)
nb_documents_classe_tea #9 documents de classe ship
proba_tea  = nb_documents_classe_tea/nb_tot_documents

#Pour calculer la probabilit� du document sachant la classe on a besoin de la probabilit� des mots (dans le document) sachant la classe

#PROBABILIT� DES MOTS SACHANT LA CLASSE

#Trade
C1 =rowSums(TF[,1:nb_documents_classe_trade])
#ACQ
a = nb_documents_classe_trade+1
b = nb_documents_classe_trade+nb_documents_classe_acq
C2 =rowSums(TF[,a:b])

#Ship
a = b+1
b = a+nb_documents_classe_ship -1
C3 =rowSums(TF[,a:b])

#Silver
a = b+1
b = a+nb_documents_classe_silver -1
C4 =rowSums(TF[,a:b])

#tea
a = b+1
b = a+nb_documents_classe_tea -1
C5 =rowSums(TF[,a:b])

#MATRICE CONTENANT L'OCCURENCE DE CHAQUE MOT DANS LES CLASSES Cj : C

lambda = 1 #Un rajoute un coefficient lambda pour �viter les probabilit�s nulles
C = cbind(C1,C2,C3,C4,C5)+lambda #C contient l'occurence des mots dans chaque classe
View(C)

#MATRICE CONTENANT LES PROBABILIT�S DES MOTS SACHANT LA CLASSE : Proba_w_sachant_C
Sum_C = colSums(C)
Proba_W_sachant_C = cbind(C[,1]/Sum_C[1],C[,2]/Sum_C[2],C[,3]/Sum_C[3],C[,4]/Sum_C[4],C[,5]/Sum_C[5])

#On a maintenant les probabilit�s des mots sachant la classe, on peut donc calculer la probabilit� du document sachant la classe
# et ainsi cr�er le classifieur bayesien qui retournera la probabilit� de chaque classe sachant le document


BayesianClassifier = function(d){
  #On met le document sous forme de corpus
  crp =  VCorpus(VectorSource(d$text))
  #Nettoyage du document
  clean_crp = cleaning(crp)
  #On met le document sous forme de sac de mot
  bow = toBagOfWords(clean_crp[[1]])
  #On restreint le sac de mots aux mots pr�sents connues gr�ce � l'apprentissage
  r = Reduce(intersect,list(names(bow)),row.names(Proba_W_sachant_C))
  #Nouveau sac de mot qui correspond au sac de mot restreint
  new_bow = bow[r]
  #Longueur du doc |d| (ici sac de mot new_bow)
  longueur_doc = as.bigz(Reduce('+',new_bow)) #as.bigz permet de manipuler par la suite des gros nombres 
  #Nombre de mots diff�rents Nd
  Nd = length(new_bow) 
  #Factoriel de la taille du doc |d|!
  longueur_doc_facto = factorial(longueur_doc)
  
  #Initilisation de liste contenant les probabilit�s d'appartenance � une classe sachant le document
  proba = list(0,0,0,0,0)
  for(i in 1:length(proba)){
    for(k in names(new_bow)){
      #Nk correspond � l'occurence du mot k dans le document
      Nk = new_bow[[k]] 
      #Application de la formule g�n�rale sur les loi multinomiales
      proba[[i]] = proba[[i]]+ log((Proba_W_sachant_C[k,i]**Nk)/factorial(Nk))
    } #REMARQUE : On passe par les logarithmes pour eviter d'effectuer des produits nen les transformant en sommes
  proba[[i]] = exp(mpfr((proba[[i]]+log(longueur_doc_facto)),precBits = 53)) #On r�cup�re les vraies valeurs gr�ce � la librairie Rmpfr 

  }
  #On divise les proba des documents sachant la classe par la somme totale des proba des documents sachant la classe pour �viter d'avoir � calculer la probabilit� des documents
  sum_proba = Reduce("+",proba)
  for(i in 1:length(proba)){
    proba[[i]] = proba[[i]]/sum_proba
  }
  
  #Traitement Post Process pour afficher proprement les diff�rentes probabilit�s des classes sachant le document ainsi que le nom de la classe la plus probable
  id = 1
  classe = list("1"="trade","2"="acq","3"="ship","4"="silver","5"="tea")
  for(i in 1:length(proba)){
    print(paste("Le document a",toString(round(as.numeric(100*proba[[i]]),4)),"% de chance d'appartenir � la classe",toString(classe[[i]])))
    if(proba[[i]]>proba[[id]]){
      id = i
    }
  }
  print(paste("Le topic le plus probable est le topic",toString(classe[[id]])))
}

#Application du classifieur bay�sien
#Exemple 1
#Sur un document test ( qui n'a pas servi � la construction du classifieur ) dont on sait qu'il appartient � la classe acq
d1 = readtext(file = "test/acq/0009613")
BayesianClassifier(d1)

#Exemple 2 
#Sur un document test ( qui n'a pas servi � la construction du classifieur ) dont on sait qu'il appartient � la classe trade
d2 = readtext(file = "test/trade/0009638")
BayesianClassifier(d2)
#La classe pr�dite est bien la classe trade

#Exemple 3 
#Sur un document test ( qui n'a pas servi � la construction du classifieur ) dont on sait qu'il appartient � une classe qui n'appartient pas aux classes du bay�sien 
d3 = readtext(file = "test/cocoa/0011132")
BayesianClassifier(d3)

#==> il pr�voit la classe trade (avec un peu moins d'assurance que pour les exemples pr�c�dents)
#==> pr�diction assez coh�rente, le th�me cocoa va plus �tre li� au th�me trade parmi la liste des 5 topics ayant �t� utilis� pour le classifieur
#==> On note quand m�me un petit pourcentage pour la classe ship, ( peut-�tre li� au transport du cacao par bateau?)

#PARTIE IV : Clustering

#Le clustering �tant un peu long, on va restreindre l'ensemble � 2 topics ship et trade

data_several_topics = c(data_ship_training$text[1:50],data_trade_training$text[1:50])
crp =  VCorpus(VectorSource(data_several_topics))
clean_crp = cleaning(crp)
#Nouvelle organisation du corpus :
#Document 1 � 197 : documents du topic "ship"
#Document 198 � 566 : documents du topic "trade

#IV.A) You first need to apply an algorithm for finding frequent termsets (A priori algorithm is recommended but not mandatory).

#On cr�e les transactions pour l'algorithme apriori : 1 transaction = 1 document

trans = c()
for(k in 1:length(clean_crp)){
  trans = c(trans,str_squish(gsub("[\r\n]", "",content(content(clean_crp)[[k]]))))
  
}
#trans contient une liste de documents "propre" ( sans les caract�res de retour � la ligne ou retour au d�but de la ligne)

#Convertion des documents en objet "transaction"
transs <- as(strsplit(trans, " "), "transactions") 
#Application de l'algorithme apriori qui va �tablir des r�gles � partir des transactions
rules = apriori(transs)

inspect(rules[1:10])

arules::itemFrequencyPlot(transs, topN = 20, 
                          col = brewer.pal(8, 'Pastel2'),
                          main = 'Relative Item Frequency Plot',
                          type = "relative",
                          ylab = "Item Frequency (Relative)")

inspect(rules)

inspect(items(rules))

ItemsSetFrequent = inspect(items(rules))[[1]]#item set frequent On a donc 2491 item fr�quent
ItemsSetFrequent


CleanItemsSetFrequent = unique(strsplit(gsub("[{} ]","",ItemsSetFrequent),",")) #On r�ecrit proprement les items set fr�quents (en enlevant les doublons au passage ) pour pouvoir travailler dessus
CleanItemsSetFrequent

#IV.B) You are asked to Compare and apply the two overlap measures (standard and entropy) defined in the paper. Can you propose another overlap measure ?

#OVERLAP MEASURE 1 : ENTROPY
#Fonction qui calcule l'entropie pour chaque item set et retourne l'item avec l'entropie minimale
entropy_overlap = function(documents,itemsSet){
  #Initialisation de f : f[j] le nombre d'itemset fr�quents pr�sent dans le document numero j
  f = numeric(length(documents)) #On initialise les fj � zero

  MD = matrix(0,length(itemsSet),length(documents)) #On initialise � z�ro la matrice qui va regrouper les coefficients d'overlap 

  #View(MD)
  for(j in 1:length(documents)){
    #On r�cup�re tous les mots du document j du corpus
    document_words = words(documents[[j]])
    
    for(i in 1:length(itemsSet)){
      if(all(itemsSet[[i]] %in% document_words)){
        #Si l'itemset est pr�sent dans le document, on incr�mente fj de 1 et on fixe Mij � 1
        f[j] = f[j]+1
        MD[i,j] = 1
      }
      
    }
  }
  #f #Pour chaque document le nombre d'item set fr�quent pr�sent dans le document
  lambda = 1 #Pour �viter les coefficients non d�finis
  f = f + lambda
  MD = MD*((-1/f)*log(1/f,exp(1)))
  #View(MD)
  MDR = rowSums(MD)#On somme les lignes de la matrice ce qui nous donne la mesure de chevauchement pour chaque item set
  ind_best_itemset = which.min(MDR) 
  return(ind_best_itemset)
}
#OVERLAP MEASURE 2 : STANDARD
#Fonction qui calcule la standardit� pour chaque item set et retourne l'item avec la valeurs minimale
standard_overlap = function(documents,itemsSet){
  #Initialisation de f : f[jl e nombre d'itemset fr�quents pr�sent dans le document numero j
  f = numeric(length(documents)) #On initialise les fj � zero
  taille_clusters = numeric(length(itemsSet))
  
  MD = matrix(0,length(itemsSet),length(documents)) #On initialise � z�ro la matrice qui va regrouper les coefficients d'overlap pour chaque
  
  #View(MD)
  for(j in 1:length(documents)){
    #On r�cup�re tous les mots du document j du corpus
    document_words = words(documents[[j]])
    
    for(i in 1:length(itemsSet)){
      if(all(itemsSet[[i]] %in% document_words)){
        #Si l'itemset est pr�sent dans le document, on incr�mente fj de 1 et on fixe Mij � 1
        f[j] = f[j]+1
        MD[i,j] = 1
        taille_clusters[i] = taille_clusters[i] + 1
      }
      
    }
  }


  #fj #Pour chaque document le nombre d'item set fr�quent pr�sent dans le document
  lambda = 1 #Pour �viter les coefficients non d�finis
  f = f + lambda
  
  
  MD = MD*(f-1)
  #View(MD)
  MDR = rowSums(MD)/taille_clusters#On somme les lignes de la matrice ce qui nous donne la mesure de chevauchement pour chaque item set
  ind_best_itemset = which.min(MDR) #Premier cluster
  
  return(itemsSet[ind_best_itemset])
}

#Probl�me standard overlap : Chaque sous set d'un item set compte comme +1 pour le fj ==> p�nalise les items set avec plusieurs items

#Fonction qui renvoie une liste de clusters : chaque cluster est repr�sent� par les num�ros des documents le composant
clustering = function(documents,itemsSet,overlap,itermax = 50) {
  indice_doc_clusters = list() #list des clusters repr�sent�s par les num�ros des documents les contenant
  ind_doc = seq(1,length(documents)) #Indice de tous les documents
  iter  = 0 #Nombre d'it�ration
  a = 1 #Num�ro du cluster cr�e
  unused_ind = seq(1,length(documents))#Indice des documents pas utilis�s
  while((length(unused_ind)>0)  & iter<itermax){
    #Recherche du meilleur item set au regard de la mesure d'overlap choisie
    print(unused_ind)
    best_itemset = overlap(documents[unused_ind],itemsSet)
    if(length(best_itemset)>0){
    print(best_itemset)
    
    indice_doc_cluster = c()
    #Ajout dans le cluster des documents pr�sentant l'item set s�lectionn�
    
    for(j in 1:length(documents)){
      document_words = words(documents[[j]])
      
      if(all(best_itemset%in%document_words)){
        indice_doc_cluster = c(indice_doc_cluster,j)
        
      }
    }
    
    if(length(indice_doc_cluster)==0){
      itemsSet = itemsSet[itemsSet != best_itemset]
    }
    else{
      #Ajout du cluster choisi � la liste des clusters 
      indice_doc_clusters[toString(a)] = list(indice_doc_cluster)
    
      unused_ind = unused_ind[!unused_ind%in%indice_doc_cluster]
      itemsSet = itemsSet[itemsSet != best_itemset]
      a = a+1
      
      print(indice_doc_cluster)
      #print(unused_ind)
      iter = iter+1
    }
    }
    else{
      itemsSet = list()
    }
    
    
  }
  
  return(indice_doc_clusters)
}

C_standard= clustering(clean_crp,CleanItemsSetFrequent,standard_overlap,500)
C_entropy= clustering(clean_crp,CleanItemsSetFrequent,entropy_overlap,500)
#On constate en effet que la mesure standard favorise les itemset de petite taille l� o� la mesure d'entropie renvoie souvent des itemsset assez grand.

#Can you propose another overlap measure ?
#Finalement le probl�me vient de la mesure de la  fr�quence, cette mesure p�nalise les items set contenant plusieurs sous item set dont chacun d'entre eux compte dans la valeur f[j]
#Id�e overlap 1 : calculer la valeur de tf idf de chaque mot des items set fr�quent dans chaque document appartenant au cluster et maximiser la somme
#En faisant �a, les mots de l'item set sont pertinents dans chacun des doc appartenant au cluster 

custom_overlap = function(documents,itemsSet){
  Is = numeric(length(itemsSet))
  taille_clusters = numeric(length(itemsSet))
  taille_itemset = numeric(length(itemsSet))
  for(j in 1:length(documents)){
    #On r�cup�re tous les mots du document j du corpus
    document_words = words(documents[[j]])
    
    for(i in 1:length(itemsSet)){
      if(all(itemsSet[[i]] %in% document_words)){
        #Si l'itemset est pr�sent dans le document, on incr�mente fj de 1 et on fixe Mij � 1
        taille_clusters[i] = taille_clusters[i] + 1
       
        for(k in 1:length(itemsSet[[i]])){
          Is[i] = Is[i] + tf_idf[itemsSet[[i]][k],j]
          taille_itemset[i] = taille_itemset[i] + 1
        }
      }
      
    }
  }
  Is = Is/(taille_clusters*taille_itemset)
  
  ind_best_itemset = which.max(Is)
  return(ind_best_itemset)

}

C_custom= clustering(clean_crp,CleanItemsSetFrequent,custom_overlap)

#Constat : Les documents sont plut�t bien r�partis dans les clusters mais les clusters sont tr�s proches ( sur les premi�res it�rations )
#Le probl�me c'est que les mots "cl�s" qui ont des valeurs de tf_idf �lev�s dominent donc tous les itemset les comportant se retrouvent s�lectionn�s en premier
#Probl�me de complexit�, quand le nombre d'itemset devient cons�quent, le code met longtemps � tourner



#Id�e overlap 2: Calculer pour chaque cluster candidat, la similarit� (avec la fonction de la premi�re partie) � l'int�rieur du corpus
#en sommant la similarit� de chaque document au centre de gravit� du cluster ainsi qu'en sommant similarit� au centre de gravit� des documents n'appartenant pas au cluster
#On fait ensuite la diff�rence des 2 sommes (Somme_interieur - Somme exterieur) et on choisit le cluster avec la valeur la plus �lev�e
#Probl�me : complexit� beaucoup trop �lev�e

# custom2_overlap = function(corpus,itemsset){
# 
#   Is = c()
# 
#   for(itemset in itemsset){
#     I = 0
#     D_int = 0
#     D_ext = 0
#     indices_membres_cluster_potentiel = c()
#     for(j in 1:length(corpus)){
#       bw = toBagOfWords(corpus[[j]])
#       if(all(itemset%in%names(bw))){#v�rifie que le document est dans la couverture de l'itemset
#         indices_membres_cluster_potentiel = c(indices_membres_cluster_potentiel,j)
# 
#       }
#     }
#     centre_gravite_int= rowMeans(tf_idf[,indices_membres_cluster_potentiel])
#     centre_gravite_ext = rowMeans(tf_idf[,-indices_membres_cluster_potentiel])
#     
# 
#     for(i in indices_membres_cluster_potentiel){
#       D_int = D_int + cos_similarity(centre_gravite_int,tf_idf_m[,j])
#       D_ext = D_ext + cos_similarity(centre_gravite_ext,tf_idf_m[,j])
# 
# 
#     }
#     I = D_ext - D_int
#    
#     Is = c(Is,I)
# 
#   }
#   ind_best_itemset = corpus[[which.max(Is)]]$meta$id
#   return(ind_best_itemset)
# 
#  }
# 
# C_custom2= clustering(clean_crp,CleanItemsSetFrequent,custom2_overlap,500)#trop long




