
### all data: https://en.wikipedia.org/wiki/Proteinogenic_amino_acid

### compute eaure vectors 


library(data.table)

aa_tab1 = read.csv2('Data/aa_mass_pi_pk.csv', stringsAsFactors = F, header = T, sep = ',')
aa_tab2 = read.csv2('Data/aa_props.csv', stringsAsFactors = F, header = T, sep = ',')
aa_tab3 = read.csv2('Data/aa_occurence.csv', stringsAsFactors = F, header = T, sep = ',')
aa_tab4 = read.csv2('Data/aa_mass.csv', stringsAsFactors = F, header = T, sep = ',')
aa_tab5 = read.csv2('Data/aa_synt_cost.csv', stringsAsFactors = F, header = T, sep = ',')
aa_tab6 = read.csv2('Data/aa_remarks.csv', stringsAsFactors = F, header = T, sep = ',')

aa_tab4$Amino.acid <- aa_tab4$Amino.Acid

aa_merged <- Reduce(function(...)
  merge(..., all = TRUE, by = c("Amino.acid", "Short", "Abbrev.")), list(aa_tab1, aa_tab2, aa_tab3[-23,], aa_tab4))

aa_merged[aa_merged==''] <- NA
aa_merged[aa_merged=='N.D.'] <- NA
aa_merged[aa_merged=='>0'] <- 0

setDT(aa_merged)
aa_merged[,basic:=0][grepl('basic', pH),basic:=1]
aa_merged[,acidic:=0][grepl('acidic', pH),acidic:=1]
aa_merged[,aromatic:=0][Aromatic.or.Aliphatic=='Aromatic',aromatic:=1]
aa_merged[,aliphatic:=0][Aromatic.or.Aliphatic=='Aliphatic',aliphatic:=1]
aa_merged[,hydrophobic:=0][Hydro..phobic=='X', hydrophobic:=1]
aa_merged[,polar:=0][Polar=='X', polar:=1]
aa_merged[,small:=0][Small=='X', small:=1]
aa_merged[,tiny:=0][Tiny=='X', tiny:=1]
aa_merged[pKa.=='-',pKa.:=NA]
setDF(aa_merged)

aa_feat <- data.frame(aa_merged[,c(2, 4:7, 10, 18:21, 25:34)])
aa_feat[is.na(aa_feat)] <- 0.   # TODO - is this correct?

for (feat in colnames(aa_feat[,-1])){
  aa_feat[[feat]] <- as.numeric(aa_feat[[feat]])
}

aa_feat[23,] <- NA
aa_feat[24,] <- NA
aa_feat[25,] <- NA

aa_feat[23,1] <- 'X'
aa_feat[23,-1] <- apply(aa_feat[-(23:25), -1], 2, mean)
aa_feat[24,1] <- 'B'
aa_feat[24,-1] <- apply(aa_feat[3:4, -1], 2, mean)
aa_feat[25,1] <- 'Z'
aa_feat[25,-1] <- apply(aa_feat[6:7, -1], 2, mean)


### compute "similarity relation" matrix 
aa_sim = matrix(0, 25, 25)
rownames(aa_sim) <- aa_feat$Short
colnames(aa_sim) <- aa_feat$Short
diag(aa_sim) <- 1

### union letters

aa_sim[c('D', 'N', 'B'), c('D', 'N', 'B')] <- 1
aa_sim[c('E', 'Q', 'Z'), c('E', 'Q', 'Z')] <- 1


### "behaves similarly", "Functionally similar to"

aa_sim[c('D', 'E'), c('D', 'E')] <- 1

aa_sim[c('K', 'R'), c('K', 'R')] <- 1

aa_sim[c('I', 'L', 'V'),c('I', 'L', 'V')] <- 1

aa_sim[c('S', 'T'), c('S', 'T')] <- 1

aa_sim[c('W', 'Y', 'F'), c('W', 'Y', 'F')] <- 1


### "Similar to", "analog"

# aa_sim[c('A', 'G'), c('A', 'G')] <- 1
# 
# aa_sim[c('N', 'D'), c('N', 'D')] <- 1
# 
# aa_sim[c('K', 'O'), c('K', 'O')] <- 1
# 
# aa_sim[c('Q', 'E'), c('Q', 'E')] <- 1
# 
# aa_sim[c('C', 'U'), c('C', 'U')] <- 1


### Hydrophobic

# aa_sim[c('W', 'Y', 'F', 'I', 'L', 'V'), c('W', 'Y', 'F', 'I', 'L', 'V')] <- 1


### write the data
write.csv(aa_feat, 'Data/aa_feat.csv')
write.csv(aa_sim, 'Data/aa_sim.csv', row.names = F)
