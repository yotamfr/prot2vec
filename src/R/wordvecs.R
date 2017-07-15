# Scatterplots
library('scatterplot3d')
library('Rtsne')

set.seed(1) # for reproducibility

(WD <- getwd())
if (!is.null(WD)) setwd(WD)

wordvecs <- read.table("./models/node2vec.128.emb", header = FALSE, stringsAsFactors=FALSE, sep = ' ')
pairwise <- read.table("./models/node2vec.128.semsim.csv", header = TRUE, stringsAsFactors=FALSE, sep = ',')

### Prepare Data ###
sample_size <- 10*1000
rix <- sample(sample_size)
labels <- pairwise[rix,]

m = sample_size
n = ncol(wordvecs)-1
vecpairs <- matrix(0, nrow=m, ncol=2*n)
for(i in 1:m){
  ix1 = which(wordvecs[,1]==labels[i,1])
  ix2 = which(wordvecs[,1]==labels[i,2])
  vecpairs[i,1:n] <- as.matrix(wordvecs[ix1,-1])
  vecpairs[i,(n+1):(2*n)] <- as.matrix(wordvecs[ix2,-1])
}
wordvecs <- wordvecs[,-1]
data <- vecpairs

#### PCA #####
wordvecs.pca <- (prcomp(wordvecs[,], center = TRUE, scale. = TRUE))
wordvecs3d <- wordvecs.pca$x

num_vecs = nrow(data)
mod = 1
r = seq(1,num_vecs,mod)

plot(wordvecs3d[r,1],wordvecs3d[r,2], main="Prot2Vec: PCA 2D Scatterplot",
     xlab="PC1", ylab="PC2", pch=20)
scatterplot3d(wordvecs3d[r,1],wordvecs3d[r,2],wordvecs3d[r,3],
              main="Prot2Vec: PCA 3D Scatterplot")

#Create a function to generate a continuous color palette
colors <- colorRampPalette(c('red','green'))(12)
#This adds a column of color values
# based on the y values
COLOR <- colors[as.numeric(cut(labels$GENEONTOLOGY,breaks = 12))]

#### t-SNE #####
tsne1 <- Rtsne(wordvecs[,], dims = 2, perplexity=30, verbose=TRUE, max_iter = 500)
plot(tsne1$Y[1:tsne1$N,1:1], tsne1$Y[1:tsne1$N,2], 
     main="Prot2Vec: tSNE 2D (singles) Scatterplot", sep="")

tsne2 <- Rtsne(data[,], dims = 2, perplexity=30, verbose=TRUE, max_iter = 500)
plot(tsne2$Y[1:tsne2$N,1:1], tsne2$Y[1:tsne2$N,2],  col = COLOR, 
     main="Prot2Vec: tSNE 2D (pairs) Scatterplot", sep="")

#### hClusterig #####

# https://www.r-bloggers.com/hierarchical-clustering-in-r-2/
# clusters <- hclust(dist(tsne$Y))
# plot(clusters)
# 
# clusterCut <- cutree(clusters, 3)
