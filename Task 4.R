library("diptest");
library("fpc");
library("pvclust");
library("cluster");
library("mclust");

data = read.csv(file="10topics.csv", header = T);
data2 = data;
data = data[,-ncol(data)];
topics = data2$topics;

# Ward Hierarchical Clustering
d = dist(data, method = "euclidean") # distance matrix
fitH = hclust(d, method="ward") 
name = as.character(data2[,ncol(data2)]);
plot(fitH, labels = name); # display dendogram
groups = cutree(fitH, k=10) # cut tree into 10 clusters
# draw dendogram with red borders around the 10 clusters 
rect.hclust(fitH, k=10, border="red");
# re-arrange the order of topics 
x = as.character(topics[fitH$order]);
a = groups[fitH$order];
table(data2[,ncol(data2)], groups)
copH = cophenetic(fitH);
cor(copH,d);

k = 2;
w = 1;
for (i in 1:9989){
  if (a[i+1] != a[i]){
    w[k] = i;
    k = k+1;
  }
}
w[11] = 9990;

top =NA;
for(i in 1:10){
    top[i] = max(x[w[i]:w[i+1]]);
}


# b = matrix(NA,10,10);
# a = table(data[,ncol(data)], groups);
# for (i in 1 : ncol(a)){
#   max = which(a[,i] == max(a[,i]));
#   b[,max] = a[,i];
# }

# KMeans
kc = kmeans(data, centers=10);
table(topics,kc$cluster);
# eucQcd = dist(data,method="euclidean");
# mds=cmdscale(eucQcd,k=2,eig=T)
# plot(mds, col=kc$cluster, main="10 Clusters");
