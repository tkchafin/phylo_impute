library(dplyr)
library(ggplot2)
library(reshape2)
library(adegenet)
library(ape)
library(stringr)
library(tibble)

setwd("/Users/tchafin/github/phylo_impute/data/test_runs/")

#function to read in structure file
read_structure <- function(x, ploidy=2){
  d<-read.table(x, header=F, sep="\t")
  dat <- read.structure(x, 
                        onerowperind=F, 
                        n.ind=(nrow(d)/ploidy), 
                        n.loc=(ncol(d)-2), 
                        col.lab=1,
                        row.marknames=0,
                        col.pop=2,
                        ask=FALSE,
                        quiet=TRUE)
  return(dat)
}


#list of files to process
#note using the structure-format because it is easier to read into genind 
myFiles <- list.files(pattern="*.imputed.str")

#process file loop
#should create list of dataframes
results <- lapply(myFiles,function(x) {
  
  #get base file names and extract simulation metadata
  x<-"p3"
  name<-gsub(".str","", gsub(".*/","",x))
  spl<-as.vector(str_split(name, "_")[[1]])
  model <-as.vector(str_split(name, "_SNP_")[[1]])[1]
  miss_data_info <- as.vector(str_split(name, "_SNP_")[[1]])[2]
  miss_type <- as.vector(str_split(miss_data_info, "_")[[1]])[1]
  miss_prop <- gsub("miss", "",  as.vector(str_split(miss_data_info, "_")[[1]])[2])
  
  #missing percentages from original simulated data
  ogphy <- paste0(paste(as.list(spl[1:length(spl)-1]), collapse="_"), ".phy")
  ogdf <- data.frame(ape::read.dna(ogphy, as.character = TRUE, format = "sequential"))
  row.names(ogdf)<-gsub("\\t","",row.names(ogdf))
  df <- ogdf %>%
    rownames_to_column(var="sample")%>% 
    dplyr::select(sample)
  df$prop<-apply(ogdf, 1, function(x) length(which(x=="n")))/ncol(ogdf)
  
  #add model metadata (needed later)
  df$file <- x
  df$model <- model
  df$miss_type <- miss_type
  df$miss_prop <- miss_prop
  
  #read imputed data
  dat<-read_structure(x)
  
  #pca
  pca<-dudi.pca(dat, nf=(length(dat$loc.fac)-1), scannf=F)
  
  #join pca loadings to metadata
  pca_df <- data.frame(pca$li) %>% rownames_to_column(var="sample")
  df_join <- dplyr::left_join(df, pca_df, by="sample")
  
  
})


#test plot
ggplot(df_join, aes(x=Axis1, y=Axis2, fill=prop, color=prop)) +
         geom_point() +
         theme_minimal()





