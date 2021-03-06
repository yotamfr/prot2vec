library(protr)
# library(GOSemSim)

ids = c("P00750", "P00751", "P00752")
prots = getUniProt(ids)
print(prots)


length(extractPAAC(prots[[1]]))

pssm <- extractPSSM(prots[[1]], database.path = "")

protseg(prots[[2]], aa = "M", k = 5)

# by GO Terms
go1 = c("GO:0005215", "GO:0005488", "GO:0005515",
        "GO:0005625", "GO:0005802", "GO:0005905")  # AP4B1
go2 = c("GO:0005515", "GO:0005634", "GO:0005681",
        "GO:0008380", "GO:0031202")                # BCAS2
go3 = c("GO:0003735", "GO:0005622", "GO:0005840",
        "GO:0006412")                              # PDE4DIP
glist = list(go1, go2, go3)
gsimmat1 = parGOSim(glist, type = "go", ont = "CC")
print(gsimmat1)


# by Entrez gene id
genelist = list(c("150", "151", "152", "1814", "1815", "1816"))
gsimmat2 = parGOSim(genelist, type = "gene")
print(gsimmat2)