

library(stringr)

args = commandArgs(trailingOnly=TRUE)

if (length(args)==0) {
  stop("At least one argument must be supplied (input Rdata file).n", call.=FALSE)
}

#load data from input file
base <- str_remove(args[1], ".RData")
load(args[1])

if (exists("rand")){
  dat <- rand
}else if (exists("biasINDV")){
  dat <- biasINDV
}else if (exists("biasPOP")){
  dat <- biasPOP
}else{
  print(paste0("No data found: ",base))
}

for(i in 1:ncol(dat)) { 
  nucs <- c("A", "T", "C", "G")
  chars <- unique(dat[,i])
  chars <- chars[!is.na(chars)]
  picked <- sample(nucs, length(chars), replace=FALSE)
  new_col <- dat[,i]
  new_col[is.na(new_col)] <- "N"
  for (j in 1:length(chars)){
    new_col[new_col==chars[j]] <- picked[j]
  }
  dat[,i]<-new_col
}

inds <- c(length(row.names(dat)) , row.names(dat))
dat2 <- apply( dat , 1 , paste , collapse = "" )
seqs <- c(nchar(unname(dat2[1])), unname(dat2))
#print(seqs)
phy <- paste0(base, ".phy")
df<-data.frame(inds, seqs)
write.table(df, file=phy, col.names=FALSE, quote=FALSE, sep="\t", row.names=FALSE)

