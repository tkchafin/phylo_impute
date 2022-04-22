

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
  picked <- sample(nucs, 2, replace=FALSE)
  new_col <- dat[,i]
  new_col[is.na(new_col)] <- "N"
  new_col[new_col==0] <- picked[1]
  new_col[new_col==2] <- picked[2]

  #NOTE: MissingData_PCA.R codes heterozygotes as 1, and homozygotes as either 0 or 2
  if ("A" %in% picked && "G" %in% picked){
    new_col[new_col==1] <- "R"
  }else if ("C" %in% picked && "T" %in% picked){
    new_col[new_col==1] <- "Y"
  }else if ("C" %in% picked && "G" %in% picked){
    new_col[new_col==1] <- "S"
  }else if ("A" %in% picked && "T" %in% picked){
    new_col[new_col==1] <- "W"
  }else if ("T" %in% picked && "G" %in% picked){
    new_col[new_col==1] <- "K"
  }else if ("A" %in% picked && "C" %in% picked){
    new_col[new_col==1] <- "M"
  }else{
    new_col[new_col==1] <- "N"
  }
  dat[,i]<-new_col
  #print(new_col)
}

inds <- c(length(row.names(dat)) , row.names(dat))
dat2 <- apply( dat , 1 , paste , collapse = "" )
seqs <- c(nchar(unname(dat2[1])), unname(dat2))
#print(seqs)
phy <- paste0(base, ".phy")
df<-data.frame(inds, seqs)
write.table(df, file=phy, col.names=FALSE, quote=FALSE, sep="\t", row.names=FALSE)

