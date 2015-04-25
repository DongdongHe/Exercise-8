install.packages("tm");
install.packages("NLP");
install.packages("openNLP");
install.packages("SnowballC");
install.packages("koRpus");
install.packages("http://datacube.wu.ac.at/src/contrib/openNLPmodels.en_1.5-1.tar.gz", repos=NULL, type = "source")
install.packages("topicmodels")
install.packages("diptest");
install.packages("fpc");
install.packages("pvclust");
install.packages("cluster");
install.packages("mclust");
install.packages("ROCR")


library("topicmodels")
library("openNLPmodels.en");
library("NLP");
library("tm");
library("openNLP");
library("SnowballC");
library("koRpus");

reuters = read.csv("reuters.csv", header = TRUE);

#remove instances with purpose = not-used.
reuters = reuters[-which(reuters$purpose == "not-used"),];

#Remove instances do no belong to any topics and remove topics do not include any docs.
x = NA;
for (i in 1:nrow(reuters)){
  x[i] = sum(reuters[i,4:(ncol(reuters)-2)]);
}
reuters = reuters[-which(x == 0),];
y = NA;
for (j in 4:(ncol(reuters)-2)){
  y[j] = sum(reuters[1:nrow(reuters),j]);
}
reuters = reuters[,-which(y == 0)];


#initialization
reuters[,"text_v"] = NA;

sent_token_annotator = Maxent_Sent_Token_Annotator();
word_token_annotator = Maxent_Word_Token_Annotator();
pos_tag_annotator = Maxent_POS_Tag_Annotator();
person_entity_annotator = Maxent_Entity_Annotator(kind = "person");
location_entity_annotator = Maxent_Entity_Annotator(kind = "location");
organization_entity_annotator = Maxent_Entity_Annotator(kind = "organization");
date_entity_annotator = Maxent_Entity_Annotator(kind = "date");
money_entity_annotator = Maxent_Entity_Annotator(kind = "money");
percentage_entity_annotator = Maxent_Entity_Annotator(kind = "percentage");

#preprocessing

for(k in 1:nrow(reuters)){
  print(k);
  #replace links, tokenize:
  data = as.String(c(as.String(reuters[k,ncol(reuters)-2]), as.String(reuters[k,ncol(reuters)-1])));
  text = data;
  text = gsub("(https?://[^/\\s]+)[^\\s]*", "LINK", text);
  text = gsub("(www?[^/\\s]+)[^\\s]*", "LINK", text);
  text = gsub("/", " ", text);
  seq = whitespace_tokenizer(text);
  text = text[seq];
  
  #Remove punctuation, :
  text = removePunctuation(text);
  text_paste = paste(text, collapse = " ")
  
  #POS tagging:
  a2 = annotate(text_paste, list(sent_token_annotator, word_token_annotator));
  
  a3 = annotate(text_paste, pos_tag_annotator, a2);
  a3w = subset(a3, type == "word");
  tags = sapply(a3w$features, "[[", "POS");
  
  #Lemmatisation:
  tagged.results = treetag(text_paste, treetagger="manual", format="obj",
                           TT.tknz=FALSE , lang="en",
                           TT.options=list(path="./TreeTagger", preset="en"))
  lemmatisation = tagged.results@TT.res$lemma;
  text[which(lemmatisation != "<unknown>" & lemmatisation != "@card@")] = lemmatisation[which(lemmatisation != "<unknown>" & lemmatisation != "@card@")];
  
  #Remove stop words:
  if(length(which(tags == "PRP" | tags == "IN" | tags == "TO" | tags == "MD" | tags == "DT" | tags == "CC" | tags == "WDT" | tags == "VBP" | tags == "PRP$")) !=0){
    text = text[-which(tags == "PRP" | tags == "IN" | tags == "TO" | tags == "MD" | tags == "DT" | tags == "CC" | tags == "WDT" | tags == "VBP" | tags == "PRP$")];
  }
  
  #NE recongition: 
  entity1 = annotate(text, list(sent_token_annotator, 
                                word_token_annotator,
                                person_entity_annotator,
                                location_entity_annotator,
                                organization_entity_annotator,
                                date_entity_annotator,
                                money_entity_annotator,
                                percentage_entity_annotator));
  entity2 = subset(entity1, type == "entity");
  if(length(entity2) !=0){
    entities = sapply(entity2$features, "[[", "kind");
    entities = toupper(entities);
    temp = as.String(text);
    for(i in 1:length(entities)){
      text = gsub(temp[entity2[i]], entities[i], text);
    }
  }
  
  #Remove Numbers
  text = gsub("[[:digit:]]+", "NUM", text);
  
  #save the text as an entry in the data frame.
  text = as.String(text);
  text = gsub(",", "", text);
  text = as.character(text);
  reuters[k,ncol(reuters)] = text;
}
write.csv(reuters, "reuters-new.csv", row.names = F);
