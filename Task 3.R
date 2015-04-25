library("topicmodels")
library("openNLPmodels.en");
library("NLP");
library("tm");
library("openNLP");
library("SnowballC");
library("koRpus");
library("e1071");
library("randomForest");
library("ROCR");


reuters = read.csv("reuters-new.csv", header = TRUE);
TFIDFMatrix = read.csv("TFIDF-terms.csv", header = TRUE);
LDAMatrix = read.csv("LDA-terms.csv", header = TRUE);
CoMatrix = read.csv("Composite-terms.csv", header = TRUE);

#select data from reuters
names = c("topic.earn", "topic.acq", "topic.money.fx", "topic.grain", "topic.crude", "topic.trade", "topic.interest", "topic.ship", "topic.wheat", "topic.corn");
matching = match(names,colnames(reuters));
data = subset(reuters,,c(3,matching));
x = NA;
for (i in 1:nrow(data)){
  x[i] = sum(data[i,2:ncol(data)]);
}
data = data[-which(x == 0),];
TFIDFMatrix = TFIDFMatrix[-which(x == 0),];
LDAMatrix = LDAMatrix[-which(x == 0),];
CoMatrix = CoMatrix[-which(x == 0),];
data = cbind(data,CoMatrix);
#data = cbind(data,LDAMatrix);
#data = cbind(data,TFIDFMatrix);
write.csv(data, "10topics.csv", row.names = F);

# make copy of topics for each instance
data2 = data;
data[,"topics"] = NA;
lr = nrow(data);
lc = ncol(data);
  for(i in 1:lr){
    for(j in 2:11){
      if(data[i,j] == 1){
        dup = data[i,];
        dup[2:11] = 0;
        dup[j] = 1;
        dup[lc] = names[j-1];
        data = rbind(data,dup);
      }
    }
    print(i);
  }
data = data[-1:-lr,];
endoftrain = max(which(data[,1] == "train"));
data = data[,-1:-11];
#colnames(data)[ncol(data)] = "class"
write.csv(data, "10topics.csv", row.names = F);


# function for k-fold
classify = function(Data, nfold, classifier, endoftrain){
  rownames = c("topic.earn", "topic.acq", "topic.money.fx", "topic.grain", "topic.crude", "topic.trade", "topic.interest", "topic.ship", "topic.wheat", "topic.corn");
  colnames = c("TP", "FN", "FP", "recall", "precision", "accuracy","fmeasure")
  nclass = length(rownames);
  parameters = matrix(0, nclass, 7, dimnames=list(rownames,colnames));
  overall = 0;
  
  
  macro.recall = 0;
  macro.precision = 0;
  micro.recall = 0;
  micro.precision = 0;
  accuracy = 0;
  Matrix = 0;
  overall = matrix(0, nclass, 7, dimnames=list(rownames,colnames));
  predict = c();
  
  for(n in 1:nfold){
    print(n);
    if (nfold == 1){
      k = (endoftrain+1) : nrow(Data);
    }else{
      k = ((nrow(Data)*(n-1)/nfold+1)) : (nrow(Data)/nfold*n);
    }
    leng = length(k);
    Data.test = Data[k, ];
    Data.train = Data[-k,];
    
    switch(classifier,
           randomForest = {
             model = randomForest(Data.train$topics~., Data.train);
             prediction = predict(model, Data.test[,-ncol(Data.test)]);
             cmatrix = table(prediction, Data.test$topics);
           },
           randomForest.ntree = {
             model = randomForest(Data.train$topics~., Data.train, ntree=50);
             prediction = predict(model, Data.test[,-ncol(Data.test)]);
             cmatrix = table(prediction, Data.test$topics);
           },
           naiveBayes = {
             model = naiveBayes(Data.train$topics~., Data.train);
             prediction = predict(model, Data.test[,-ncol(Data.test)]);
             cmatrix = table(prediction, Data.test$topics);
           },
           SVM.linear = {
             model = svm(Data.train$topics~., Data.train, kernel = "linear", scale = F);
             prediction = predict(model, Data.test[,-ncol(Data.test)]);
             cmatrix = table(prediction, Data.test$topics);
           },
           SVM.Gaussian = {
             model = svm(Data.train$topics~., Data.train, scale = F);
             prediction = predict(model, Data.test[,-ncol(Data.test)]);
             cmatrix = table(prediction, Data.test$topics);
           },
           SVM.polynomial = {
             model = svm(Data.train$topics~., Data.train, kernel = "polynomial", scale = F);
             prediction = predict(model, Data.test[,-ncol(Data.test)]);
             cmatrix = table(prediction, Data.test$topics);
           }
    )
    
    for(i in 1:nclass){
      parameters[i,1] = cmatrix[i,i];
      parameters[i,2] = sum(cmatrix[-i,i], na.rm = TRUE);
      parameters[i,3] = sum(cmatrix[i,-i], na.rm = TRUE);
      parameters[i,4] = parameters[i,1]/(sum(parameters[i,1:2], na.rm = TRUE));
      parameters[i,5] = parameters[i,1]/(sum(parameters[i,c(1,3)], na.rm = TRUE));
      parameters[i,6] = parameters[i,1]/leng;
      parameters[i,7] = sum(2*parameters[i,4]*parameters[i,5], na.rm = TRUE)/(sum(parameters[i,4:5],na.rm = TRUE));
    }
    macro.recall[n] = mean(parameters[,4], na.rm = TRUE);
    macro.precision[n] = mean(parameters[,5], na.rm = TRUE);
    micro.recall[n] = sum(parameters[,1], na.rm = TRUE)/sum(parameters[,1:2], na.rm = TRUE); 
    micro.precision[n] = micro.recall[n];
    accuracy[n] = sum(parameters[,6], na.rm = TRUE);
    Matrix = Matrix + cmatrix;
    overall = overall + parameters;
    #print(parameters);
    prediction = as.character(prediction);
    predict = c(predict,prediction);
  }
  
  for(i in 1:nclass){
    overall[i,4] = overall[i,1]/(sum(overall[i,1:2], na.rm = TRUE));
    overall[i,5] = overall[i,1]/(sum(overall[i,c(1,3)], na.rm = TRUE));
    overall[i,6] = overall[i,1]/(sum(overall[,1:2], na.rm = TRUE));
    overall[i,7] = 2*overall[i,4]*overall[i,5]/(sum(overall[i,4:5],na.rm = TRUE));
  }
  macro.recall[nfold+1] = mean(overall[,4], na.rm = TRUE);
  macro.precision[nfold+1] = mean(overall[,5], na.rm = TRUE);
  micro.recall[nfold+1] = sum(overall[,1], na.rm = TRUE)/sum(overall[,1:2], na.rm = TRUE); 
  micro.precision[nfold+1] = micro.recall[nfold+1];
  accuracy[nfold+1] = sum(overall[,6], na.rm = TRUE);
  
  cat("\n Overall Parameters: ");
  cat("\n");
  print(overall);
  cat("\n Macro Recall: ", macro.recall);
  cat("\n Macro precision: ", macro.precision);
  cat("\n Micro Recall: ", micro.recall);
  cat("\n Micro precision: ", micro.precision);
  cat("\n Accuracy: ", accuracy);
  cat("\n Confusion Matrix: ");
  cat("\n");
  print(Matrix);
  return(overall);
}


# 1 fold is for the original data(train for training and test for predition)
data = read.csv("10topics.csv", header = TRUE);

randomForest = classify(data, 1, "randomForest", endoftrain);
randomForest.ntree = classify(data, 1, "randomForest.ntree", endoftrain);
naiveBayes = classify(data, 1, "naiveBayes", endoftrain);
SVM.Gaussian = classify(data, 1, "SVM.Gaussian", endoftrain);
SVM.linear = classify(data, 1, "SVM.linear", endoftrain);
SVM.polynomial = classify(data, 1, "SVM.polynomial", endoftrain);


# run the classifer for 10-fold cross validation
data = read.csv("10topics.csv", header = TRUE);
random = sample(1:nrow(data), nrow(data), replace = FALSE);
data = data[random,];

randomForest = classify(data, 10, "randomForest",);
randomForest.ntree = classify(data, 10, "randomForest.ntree",);
naiveBayes = classify(data, 10, "naiveBayes",);
SVM.Gaussian = classify(data, 10, "SVM.Gaussian",);
SVM.linear = classify(data, 10, "SVM.linear",);
SVM.polynomial = classify(data, 10, "SVM.polynomial",);
