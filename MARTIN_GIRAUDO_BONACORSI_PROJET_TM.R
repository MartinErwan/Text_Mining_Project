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

#We notice that words "trade","said","japan","japanese","washington"
#So we have a certain similarity between the wordcloud of the training set and the wordcloud of the testing set

#NB: Pictures are saved in the folder for a better vizualisation.

#Wordcloud example 2 : Topic ship

data_ship_training = readtext(file = "training/ship/*")
data_ship_test = readtext(file = "test/ship/*")

#Training set wordcloud
wc(data_ship_training)

#Testing set wordcloud
wc(data_ship_test)

#We notice that the words "trade", "said", "japan", "japanese", "washington"
#So we have some similarity between the wordcloud of the training set and the testing set

#NOTE: The images are saved in the folder for better visualization
#We notice again the recurrence of words in the 2 sets: vessels,said,spokesman,ship,foreign...

#==> We can then say that the occurrence of some words can allow to determinate the class of a document
#II.C) Using a subset of your data, represent each document as a vector of TF-IDF values and 
#use those vectors to measure similarity between pairs of documents.

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
#NOTE : The wordLengths allows to consider words lower than 3 letter. It solves some problems in the part IV of the project.

TF = t(as.matrix(dtm))
#View(TF) #The column j corresponds to the vector TF of the document j on all words of the corpus

#Step 2 : IDF matrix creation

nb_doc = ncol(TF) #We retrieve the number of documents in the corpus
#Creation of the binary matrix such that the coefficient is 0 if the word does not appear in the document and 1 otherwise
presence_doc = as.matrix(dtm) #It corresponds to the Term Frequency matrix before its transposition
presence_doc[presence_doc>0]=1
#Sum over the columns to get the number of documents which contain the word in question
frequency = colSums(presence_doc) 
#IDF
idf = log(nb_doc/frequency,2) #Idf associated to the term w of the corpus idf

#Matrix TF_IDF
tf_idf = idf*TF
#Conversion of tf_if into a dataframe to simplify its use
tf_idf = as.data.frame(tf_idf)
#Column j corresponds to the TF-IDF vector of document j.
#View(tf_idf)

#Similarity between 2 documents

#Function that, knowing the vectors tf_idf of the pair of documents, will calculate the similarity by giving a value between 0 and 1
#Knowing that 0 means that the two documents are very little similar and 1 that the documents are very similar

cos_similarity = function(v1,v2){
  norm_d1 = sqrt(dot(v1,v1))
  norm_d2 = sqrt(dot(v2,v2))
  ps = dot(v1,v2)
  
  return((ps)/(norm_d1*norm_d2))
}
#Conversion of the matrix tf_idf into a matrix object to be able to do scalar products and other calculations on the columns
tf_idf_m = as.matrix(tf_idf)

#Example 1: Similarity of a document with itself

#Retrieval of the tf_idf vector of the document (here the document number 1 of the corpus)
v1 = tf_idf_m[,1]
similarity_example1 = cos_similarity(v1,v1)
similarity_example1 #We find 1

#Example 2 : Similarity of a document with a document belonging to the same class 

#Retrieval of tf_idf vectors of documents to compare
#Here we will compare document 1 to document 2 (which belongs to the same class "trade" according to the construction of the corpus (the first 369 are of the class trade)
v2 = tf_idf_m[,2]
similarity_example2 = cos_similarity(v1,v2)
similarity_example2 #We find 0.11

#Exemple 3 : Similarity of a document with a document belonging to a different class
#Retrieval of tf_idf vector of the document (Here the document number 2000 of the corpus so a document of the class acq)
v3 = tf_idf_m[,2000]
similarity_example3 = cos_similarity(v1,v3)
similarity_example3 #On trouve 0.0008==> valeur encore plus faible que si les documents appartenaient � la meme classe

#PART III Classification : Create and test a Bayesian classifier for your data.

#III.A) You first need to represent each document as a bag of words.

#Function to convert a document into a bag of word
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
#Example of application on the first document of the corpus 
d1 = clean_crp[[1]]
content(d1)
bw_d1 = toBagOfWords(d1)
bw_d1

#III.B) Classification will be applied to (a) subset(s) of the data containing less than 10 topics.

#We will do classification on the 5 topics of the part II

#Classes : trade, acq, ship, silver, tea

#Reminder of the organization of the corpus 
# Documents 1 to 369 : Topic "Trade"
# Documents 370 to 2019 : Topic "acq"
# Documents 2020 to 2216 : Topic "ship"
# Documents 2217 to 2237 : Topic "silver"
# Documents 2238 to 2246 : Topic "tea"

#To create the classifier, we need the probability of the classes, the probability of the document knowing the class and the probability of the document

#Class probabilities 

nb_tot_documents= length(data_several_topics) 
nb_tot_documents#2246 documents in total in the training set

#trade probability
nb_documents_classe_trade = length(data_trade_training$text) #369 documents de la classe trade
nb_documents_classe_trade
proba_trade  = nb_documents_classe_trade/nb_tot_documents

#acq probability
nb_documents_classe_acq = length(data_acq_training$text)
nb_documents_classe_acq #1650 documents de classe acq
proba_acq  = nb_documents_classe_acq/nb_tot_documents

#ship probability
nb_documents_classe_ship = length(data_ship_training$text)
nb_documents_classe_ship #197 documents de classe ship
proba_ship  = nb_documents_classe_ship/nb_tot_documents

#silver probability
nb_documents_classe_silver = length(data_silver_training$text)
nb_documents_classe_silver #21 documents de classe ship
proba_silver  = nb_documents_classe_silver/nb_tot_documents

#tea probability
nb_documents_classe_tea = length(data_tea_training$text)
nb_documents_classe_tea #9 documents de classe ship
proba_tea  = nb_documents_classe_tea/nb_tot_documents

#To calculate the probability of the document knowing the class we need the probability of the words (in the document) knowing the class

#PROBABILITY OF WORDS KNOWING THE CLASS

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

#MATRICE CONTAINING THE OCCURENCE OF EACH WORD IN THE Cj CLASSES: C

lambda = 1 #We add a lambda coefficient to avoid zero probabilities
C = cbind(C1,C2,C3,C4,C5)+lambda #C contains the occurence of words in each class 
View(C)

#MATRICE CONTAINING THE PROBABILITIES OF WORDS KNOWING THE CLASS: Proba_w_sachant_C
Sum_C = colSums(C)
Proba_W_sachant_C = cbind(C[,1]/Sum_C[1],C[,2]/Sum_C[2],C[,3]/Sum_C[3],C[,4]/Sum_C[4],C[,5]/Sum_C[5])

#We have now probabilities of words knowing the class, so we can calculate the document probability knowing the class 
# and thus create the Bayesian classifier that will return the probability of each class knowing the document


BayesianClassifier = function(d){
  #We put the document in the form of a corpus
  crp =  VCorpus(VectorSource(d$text))
  #Document cleaning
  clean_crp = cleaning(crp)
  #We put the document in the form of a bag of words
  bow = toBagOfWords(clean_crp[[1]])
  #We restrict the bag of words to the present words known through learning.
  r = Reduce(intersect,list(names(bow)),row.names(Proba_W_sachant_C))
  #New word bag that matches the restricted word bag
  new_bow = bow[r]
  #Document length |d| (here bag of words : new_bow)
  longueur_doc = as.bigz(Reduce('+',new_bow)) #as.bigz allows you to manipulate large numbers afterwards 
  #Number of differents words Nd
  Nd = length(new_bow) 
  #Factorial of the size of the doc |d|!
  longueur_doc_facto = factorial(longueur_doc)
  
  #Initialization of a list containing the probabilities of belonging to a class knowing the document
  proba = list(0,0,0,0,0)
  for(i in 1:length(proba)){
    for(k in names(new_bow)){
      #Nk corresponds to the occurrence of word k in the document
      Nk = new_bow[[k]] 
      #Application of the general formula on multinomial laws
      proba[[i]] = proba[[i]]+ log((Proba_W_sachant_C[k,i]**Nk)/factorial(Nk))
    } #NOTE: We use logarithms to avoid making products by transforming them into sums
  proba[[i]] = exp(mpfr((proba[[i]]+log(longueur_doc_facto)),precBits = 53)) # #We retrieve the real values thanks to the Rmpfr library 

  }
  #We divide the probabilities of the documents knowing the class by the sum total of the probabilities of the documents knowing the class to avoid having to calculate the probability of the documents
  sum_proba = Reduce("+",proba)
  for(i in 1:length(proba)){
    proba[[i]] = proba[[i]]/sum_proba
  }
  
  #Post Processing to cleanly display the different probabilities of the classes knowing the document as well as the name of the most likely class
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

#Application of the bayesian classifier
#Example 1
#On test document ( that was not used to build the classifier ) which is known to belong to the class acq
d1 = readtext(file = "test/acq/0009613")
BayesianClassifier(d1)

#Example 2 
#On test document ( that was not used to build the classifier ) which is known to belong to the class trade
d2 = readtext(file = "test/trade/0009638")
BayesianClassifier(d2)
#The predicted class is indeed the trade class

#Example 3 
#On test document ( that was not used to build the classifier ) which is known to belong to a class that does not belong to the Bayesian class 
d3 = readtext(file = "test/cocoa/0011132")
BayesianClassifier(d3)

#==> it foresees the trade class (with a little less confidence than for the previous examples)
#==> fairly consistent prediction, the cocoa topic is going to be more related to the trade topic among the list of 5 topics having been used for the classifier
#==> We note however a small percentage for the ship class, (perhaps related to the transport of cocoa by boat?)

#PART IV : Clustering

#The clustering being a bit long, we will restrict the set to 2 topics ship and trade

data_several_topics = c(data_ship_training$text[1:50],data_trade_training$text[1:50])
crp =  VCorpus(VectorSource(data_several_topics))
clean_crp = cleaning(crp)
#New corpus' organization:
#Document 1 to 197 :  "ship" topic documents
#Document 198 to 566 : "trade" topic documents

#IV.A) You first need to apply an algorithm for finding frequent termsets (A priori algorithm is recommended but not mandatory).

#We create the transactions for the apriori algorithm: 1 transaction = 1 document

trans = c()
for(k in 1:length(clean_crp)){
  trans = c(trans,str_squish(gsub("[\r\n]", "",content(content(clean_crp)[[k]]))))
  
}
#trans contains a list of "clean" documents ( without \n or \r ) 

#Converting documents into "transaction" objects
transs <- as(strsplit(trans, " "), "transactions") 
#Application of the apriori algorithm that will establish rules from the transactions
rules = apriori(transs)

inspect(rules[1:10])

arules::itemFrequencyPlot(transs, topN = 20, 
                          col = brewer.pal(8, 'Pastel2'),
                          main = 'Relative Item Frequency Plot',
                          type = "relative",
                          ylab = "Item Frequency (Relative)")

inspect(rules)

inspect(items(rules))

ItemsSetFrequent = inspect(items(rules))[[1]]#item set frequent So we have 2491 frequent items
ItemsSetFrequent


CleanItemsSetFrequent = unique(strsplit(gsub("[{} ]","",ItemsSetFrequent),",")) #We rewrite the frequent set items (removing duplicates in the process) to be able to work on them
CleanItemsSetFrequent

#IV.B) You are asked to Compare and apply the two overlap measures (standard and entropy) defined in the paper. Can you propose another overlap measure ?

#OVERLAP MEASURE 1 : ENTROPY
#Function that calculates the entropy for each item set and returns the item with the minimum entropy
entropy_overlap = function(documents,itemsSet){
  #Initialization of f : f[j] the number of frequent itemset present in the document number j
  f = numeric(length(documents)) #We initialize the fj to zero

  MD = matrix(0,length(itemsSet),length(documents)) #We initialize to zero the matrix that will group the overlap coefficients 

  #View(MD)
  for(j in 1:length(documents)){
    #We retrieve all the words of document j in the corpus
    document_words = words(documents[[j]])
    
    for(i in 1:length(itemsSet)){
      if(all(itemsSet[[i]] %in% document_words)){
        #If the itemset is present in the document, we increment fj by 1 and set Mij to 1
        f[j] = f[j]+1
        MD[i,j] = 1
      }
      
    }
  }
  #For each document the number of frequent set items present in the document
  lambda = 1 #To avoid undefine coefficient
  f = f + lambda
  MD = MD*((-1/f)*log(1/f,exp(1)))
  #View(MD)
  MDR = rowSums(MD)#We sum the rows of the matrix which gives us the overlap measure for each item set
  ind_best_itemset = which.min(MDR) 
  return(ind_best_itemset)
}
#OVERLAP MEASURE 2 : STANDARD
#Function that calculates the standardity for each item set and returns the item with the minimum value
standard_overlap = function(documents,itemsSet){
  #Initialization of f : f[jl e number of itemset frequents present in the document numero j
  f = numeric(length(documents)) #We initialize the fj to zero
  taille_clusters = numeric(length(itemsSet))
  
  MD = matrix(0,length(itemsSet),length(documents)) #We initialize to zero the matrix that will group the overlap coefficients for each
  
  #View(MD)
  for(j in 1:length(documents)){
    #We retrieve all the words of document j from the corpus
    document_words = words(documents[[j]])
    
    for(i in 1:length(itemsSet)){
      if(all(itemsSet[[i]] %in% document_words)){
        #If the itemset is present in the document, we increment fj by 1 and set Mij to 1
        f[j] = f[j]+1
        MD[i,j] = 1
        taille_clusters[i] = taille_clusters[i] + 1
      }
      
    }
  }


  #fj #For each document the number of frequent set items present in the document
  lambda = 1 #To avoid undefined coefficients
  f = f + lambda
  
  
  MD = MD*(f-1)
  #View(MD)
  MDR = rowSums(MD)/taille_clusters#We sum the rows of the matrix which gives us the overlap measure for each set item
  ind_best_itemset = which.min(MDR) #First cluster
  
  return(itemsSet[ind_best_itemset])
}

#Standard overlap problem : Each subset of an item set counts as +1 for the fj ==> penalize items set with several items

#Function that returns a list of clusters: each cluster is represented by the numbers of the documents in it
clustering = function(documents,itemsSet,overlap,itermax = 50) {
  indice_doc_clusters = list() #list of clusters represented by the numbers of the documents containing them
  ind_doc = seq(1,length(documents)) #Index of all documents
  iter  = 0 #Number of iterations
  a = 1 #Number of the cluster created
  unused_ind = seq(1,length(documents))#Index of documents not used
  while((length(unused_ind)>0)  & iter<itermax){
    #Search for the best item set according to the chosen overlap measure
    print(unused_ind)
    best_itemset = overlap(documents[unused_ind],itemsSet)
    if(length(best_itemset)>0){
    print(best_itemset)
    
    indice_doc_cluster = c()
    #Add the documents with the selected item set to the cluster
    
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
      #Add the selected cluster to the list of clusters 
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
#Indeed, we notice that the standard measure favors small itemset while the entropy measure often returns quite large itemset.




