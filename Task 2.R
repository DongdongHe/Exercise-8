library("topicmodels")
library("openNLPmodels.en");
library("NLP");
library("tm");
library("openNLP");
library("SnowballC");
library("koRpus");

reuters = read.csv("reuters-new.csv", header = TRUE);

#DocumentTermMatrix
text = Corpus(VectorSource(reuters$text_v));
text = tm_map(text, PlainTextDocument);
dtm = DocumentTermMatrix(text);

#TFIDF
dtm2 = removeSparseTerms(dtm,0.9);
##############################
# rowTotals = apply(dtm2,1,sum);
# dtm3 = dtm2[rowTotals>0,];
# reuters = reuters[-which(rowTotals==0),];
# reuters = cbind(reuters,inspect(dtm3));
TFMatrix = as.data.frame(inspect(dtm2));
rownames(TFMatrix) = c(1:nrow(TFMatrix));
TFIDFMatrix = TFMatrix;
D = nrow(TFMatrix);
for (j in 1:ncol(TFMatrix)){
  DF = length(which(TFMatrix[,j]>0));
  maxF = max(TFMatrix[,j]);
  quotient = D/DF;
  for (i in 1:nrow(TFMatrix)){
    TF = TFMatrix[i,j]/maxF;
    IDF = log(2, quotient);
    TFIDFMatrix[i,j] = TF*IDF;
  }
  print(j);
}

Matrixbackup = TFIDFMatrix;

TFIDFMatrix = Matrixbackup;

for(i in 1:nrow(TFIDFMatrix)){
  pick = which(TFIDFMatrix[i,] !=0);
  a = length(pick);
  if(a > 20){
    threshold = order(as.numeric(TFIDFMatrix[i,pick]))[20];
  }else{
    threshold = 0.000001;
  }
  TFIDFMatrix[i, which(TFIDFMatrix[i,] >= threshold)] = 1;
  print(i);
}
TFIDFMatrix = floor(TFIDFMatrix);
sum = apply(TFIDFMatrix,2,sum);
write.csv(TFIDFMatrix, "TFIDF-terms.csv", row.names = F);
#colTotals = apply(TFIDFMatrix,2,sum);
#rowTotals = apply(TFIDFMatrix,1,sum);



#LDA topic models
vem<-LDA(dtm,method="VEM",k=10);
vem.terms = terms(vem,10);
uterms = unique(as.vector(vem.terms))
#load(LDA.RData);

#LDA terms frequency
M = matrix(NA, 10, length(uterms));
LDAMatrix = as.data.frame(matrix(NA, length(topics(vem)), length(uterms)));
colnames(LDAMatrix) = c(uterms);
for (i in 1:10){
  f = match(uterms, vem.terms[,i]);
  f[which(is.na(f)==FALSE)] = 1;
  f[which(is.na(f)==TRUE)] = 0;
  M[i,] = f;
}
topiclist = topics(vem);
for(i in 1: nrow(LDAMatrix)){
  #text i belongs to topic t
  t = topiclist[i];
  LDAMatrix[i,] = M[t,];
  print(i);
}
write.csv(LDAMatrix, "LDA-terms.csv", row.names = F);



#composite of LDA and TFIDF
CoMatrix = TFIDFMatrix;
matching = match(colnames(LDAMatrix),colnames(TFIDFMatrix));
rownames(CoMatrix) = rownames(LDAMatrix);
for(i in 1:ncol(LDAMatrix)){
  if(is.na(matching[i]) == FALSE){
    CoMatrix[,matching[i]] = as.numeric(TFIDFMatrix[,matching[i]] |  LDAMatrix[,i]);
  }
  else{
    CoMatrix = cbind(CoMatrix,LDAMatrix[,i]);
    colnames(CoMatrix)[ncol(CoMatrix)] = colnames(LDAMatrix)[i];
  }
}
write.csv(CoMatrix, "Composite-terms.csv", row.names = F);
