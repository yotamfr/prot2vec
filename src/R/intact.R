library(RpsiXML)
library(mongolite)

xmlDir <- system.file("data/IntAct/all/psi25/species",package="RpsiXML")
intactxml <- file.path(xmlDir, "intact_2008_test.xml")