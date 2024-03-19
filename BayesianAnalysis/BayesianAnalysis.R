setwd(normalizePath(dirname(rstudioapi::getActiveDocumentContext()$path)))
df <- read.csv('data_for_R.csv')

# BEST package is no longer active, but can still be downloaded from CRAN archive
# Download package tarball from CRAN archive
#url <- "https://cran.r-project.org/src/contrib/Archive/BEST/BEST_0.5.4.tar.gz"
#pkgFile <- "BEST_0.5.4.tar.gz"
#download.file(url = url, destfile = pkgFile)
# make sure the dependencies coda and rjags are installed
# Install package from downloaded file
#install.packages(pkgs=pkgFile, type="source", repos=NULL)

library(BEST)
library(tidyverse)

agg_context <- df %>% 
  summarize(mean = mean(context),
            sd = sd(context))

agg_concept <- df %>%
  summarize(mean = mean(concept),
            sd = sd(concept))

grand_mean <- (agg_context$mean + agg_concept$mean)/2
grand_sd <- (agg_context$sd + agg_concept$sd)/2

# calculate a region of practical equivalence with zero according to recommendation by Kruschke (2018)
rope <- c(-0.1*grand_sd, 0.1*grand_sd)

context_aware_generic_concepts <- df %>% 
  filter(index == 0, condition == 'context_aware') %>% 
  select(index, concept, condition)

context_unaware_generic_concepts <- df %>% 
  filter(index == 0, condition == 'context_unaware') %>% 
  select(index, concept, condition)

context_aware_specific_concepts <- df %>% 
  filter(index == 4, condition == 'context_aware') %>% 
  select(index, concept, condition) 

context_unaware_specific_concepts <- df %>% 
  filter(index == 4, condition == 'context_unaware') %>% 
  select(index, concept, condition) 

context_aware_coarse_contexts <- df %>% 
  filter(index == 0, condition == 'context_aware') %>% 
  select(index, context, condition)

context_unaware_coarse_contexts <- df %>% 
  filter(index == 0, condition == 'context_unaware') %>% 
  select(index, context, condition)

context_aware_fine_contexts <- df %>% 
  filter(index == 4, condition == 'context_aware') %>% 
  select(index, context, condition)

context_unaware_fine_contexts <- df %>% 
  filter(index == 4, condition == 'context_unaware') %>% 
  select(index, context, condition)


priors <- list(muM = grand_mean, muSD = grand_sd)

# either load or generate models
#load("BESTgeneric.Rda")
#load("BESTspecific.Rda")
#load("BESTcoarse.Rda")
#load("BESTfine.Rda")
BESTgeneric <- BESTmcmc(context_unaware_generic_concepts$concept, context_aware_generic_concepts$concept, priors=priors, parallel=TRUE)
BESTspecific <- BESTmcmc(context_unaware_specific_concepts$concept, context_aware_specific_concepts$concept, priors=priors, parallel=TRUE)
BESTcoarse <- BESTmcmc(context_unaware_coarse_contexts$context, context_aware_coarse_contexts$context, priors=priors, parallel=TRUE)
BESTfine <- BESTmcmc(context_unaware_fine_contexts$context, context_aware_fine_contexts$context, priors=priors, parallel=TRUE)

# check for convergence
print(BESTgeneric)
print(BESTspecific)
print(BESTcoarse)
print(BESTfine)
# -> all models converged

Diff_generic <- (BESTgeneric$mu1 - BESTgeneric$mu2)
meanDiff_generic <- round(mean(Diff_generic), 3)
hdiDiff_generic <- hdi(BESTgeneric$mu1 - BESTgeneric$mu2)
plotAll(BESTgeneric)
plot(BESTgeneric,ROPE=rope)
summary(BESTgeneric)
# CrI includes 0
# 95.9% probability that the difference in means is larger than 0 (pd)
# 6% in ROPE

Diff_specific <- (BESTspecific$mu1 - BESTspecific$mu2)
meanDiff_specific <- round(mean(Diff_specific), 3)
hdiDiff_specific <- hdi(BESTspecific$mu1 - BESTspecific$mu2)
plotAll(BESTspecific)
plot(BESTspecific, ROPE=rope)
summary(BESTspecific)
# CrI doesn not include 0
# 99.4% probability that the difference in means is larger than 0
# 0% in ROPE

Diff_coarse <- (BESTcoarse$mu1 - BESTcoarse$mu2)
meanDiff_coarse <- round(mean(Diff_coarse), 3)
hdiDiff_coarse <- hdi(BESTcoarse$mu1 - BESTcoarse$mu2)
plotAll(BESTcoarse)
plot(BESTcoarse, ROPE=rope)
summary(BESTcoarse)
# CrI does not include 0
# 100% probability that the difference in means is larger than 0
# 0% in ROPE

Diff_fine <- (BESTfine$mu1 - BESTfine$mu2)
meanDiff_fine <- round(mean(Diff_fine), 3)
hdiDiff_fine <- hdi(BESTfine$mu1 - BESTfine$mu2)
plotAll(BESTfine)
plot(BESTfine, ROPE=rope)
summary(BESTfine)
# CrI includes 0
# 70.1% probability that the difference in means is larger than 0
# 20% in ROPE

# save all models for reproducibility
write.csv(BESTgeneric, "BESTgeneric.csv", row.names=FALSE, quote=FALSE) 
save(BESTgeneric,file="BESTgeneric.Rda")
write.csv(BESTspecific, "BESTspecific.csv", row.names=FALSE, quote=FALSE) 
save(BESTspecific,file="BESTspecific.Rda")
write.csv(BESTgeneric, "BESTcoarse.csv", row.names=FALSE, quote=FALSE) 
save(BESTgeneric,file="BESTcoarse.Rda")
write.csv(BESTgeneric, "BESTfine.csv", row.names=FALSE, quote=FALSE) 
save(BESTgeneric,file="BESTfine.Rda")
