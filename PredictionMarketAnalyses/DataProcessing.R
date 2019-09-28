##################################################################################
### NEUROIMAGING ANALYSIS AND REPLICATION PROJECT (NARPS) - PREDICTION MARKETS ###
### -------------------------------------------------------------------------- ###
### ---------------------------- Data Processing ----------------------------- ###
##################################################################################



basedir = Sys.getenv('NARPS_BASEDIR')
if (basedir == ""){
  # use default
  basedir = "/data"
}

use_checkpoint = TRUE

s = Sys.info()
if (s['sysname'] == 'Darwin'){
  use_checkpoint = FALSE
}

if (use_checkpoint) {
  library(checkpoint)
  checkpointDir <- '/checkpoint'
  checkpoint("2019-08-13", checkpointLocation = checkpointDir)
  }


# Libraries #
# ~~~~~~~~~ #
library(fs)
library(dplyr)
library(tidyr)
library(lubridate)

# Working Directories #
# ~~~~~~~~~~~~~~~~~~~ #

# setup orig dirs
raw <- paste(basedir,'orig/PredictionMarkets',sep='/')
nonTeam <- paste(raw, 'NonTeamMembers',sep='/')
team <- paste(raw, 'TeamMembers',sep='/')

# setup derived data dirs
output_dir <- paste(basedir,'PredictionMarkets',sep='/')
if (!dir.exists(output_dir)){
  dir.create(output_dir)
}
comb <- paste(output_dir, 'Combined',sep='/')
if (!dir.exists(comb)){
  dir.create(comb)
}
proc <- paste(output_dir, 'Processed',sep='/')
if (!dir.exists(proc)){
  dir.create(proc)
}

### Import Raw Data                                                            ###
### -------------------------------------------------------------------------- ###

# Non-Team Members Data #
# ~~~~~~~~~~~~~~~~~~~~~ #

nt_asset <- read.csv(paste(nonTeam, 'asset.csv', sep='/' ))
nt_portfolio <- read.csv(paste(nonTeam, 'portfolio.csv', sep='/' ))
nt_position <- read.csv(paste(nonTeam, 'position.csv', sep='/' ))
nt_transaction <- read.csv(paste(nonTeam, 'transaction.csv', sep='/' ))

nt_asset$teams <- 0
nt_portfolio$teams <- 0
nt_position$teams <- 0
nt_transaction$teams <- 0
  
names(nt_asset)[1] <- "asset_id"
names(nt_portfolio)[1] <- "portfolio_id"
names(nt_position)[1] <- "position_id"
names(nt_transaction)[1] <- "transaction_id"
  


# Team Members Data #
# ~~~~~~~~~~~~~~~~~ #


t_asset <- read.csv(paste(team, 'asset.csv', sep='/' ))
t_portfolio <- read.csv(paste(team, 'portfolio.csv', sep='/' ))
t_position <- read.csv(paste(team, 'position.csv', sep='/' ))
t_transaction <- read.csv(paste(team, 'transaction.csv', sep='/' ))

t_asset$teams <- 1
t_portfolio$teams <- 1
t_position$teams <- 1
t_transaction$teams <- 1

names(t_asset)[1] <- "asset_id"
names(t_portfolio)[1] <- "portfolio_id"
names(t_position)[1] <- "position_id"
names(t_transaction)[1] <- "transaction_id"



# Append Data from Non-Team and Team Members #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

asset <- rbind(nt_asset,t_asset)
portfolio <- rbind(nt_portfolio,t_portfolio)
position <- rbind(nt_position,t_position)
transaction <- rbind(nt_transaction,t_transaction)

save(asset, 
     file = paste(comb,"asset.RData", sep='/'))
save(portfolio, 
     file = paste(comb,"portfolio.RData", sep='/'))
save(position,
     file = paste(comb,"position.RData", sep='/'))
save(transaction,
     file = paste(comb,"transaction.RData", sep='/'))
    
    
# Import Team Results Data #
# ~~~~~~~~~~~~~~~~~~~~~~~~ # 

TeamResults<-read.csv(
  file = paste(raw, "TeamResults.csv", sep='/'),
  header = TRUE, 
  sep = ","
)
TeamResults$teams <- 1
names(TeamResults)[1] <- "teamresults_id"

save(TeamResults, 
       file = paste(comb,"teamresults.RData", sep='/'))
    

  
### Clean-Up Data Tabels                                                       ###
### -------------------------------------------------------------------------- ###

# Clean-Up Data Asset Data Table #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
asset$market_id <- NULL
asset$type <- NULL
asset$description <- NULL

# Rename Variables
names(asset)[names(asset) == "shares_outcome1"] <- "long_outstanding"
names(asset)[names(asset) == "shares_outcome2"] <- "short_outstanding"

# Hypothesis ID  
asset$hid <- asset %>% group_indices(asset_id) 
asset$hid <- asset$hid - asset$teams * 9
asset$asset_id <- NULL

# Hypothesis
asset$hypothesis <- substring(asset$name,6)
asset$name <- NULL

# Order and Sort
asset <- asset[c("teams","hid","hypothesis","long_outstanding","short_outstanding","price")]
asset <- asset[order(asset$teams,asset$hid),]

# Save
save(asset,
     file = paste(comb, "asset.RData", sep='/'))
    
    
# Clean-Up Data Portfolio Data Table #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
portfolio <- subset(portfolio,
                    select=c(portfolio_id, 
                             trader_id,
                             balance, 
                             teams))

# Rename Variables
names(portfolio)[names(portfolio) == "portfolio_id"] <- "aid"
names(portfolio)[names(portfolio) == "trader_id"] <- "uid"
names(portfolio)[names(portfolio) == "balance"] <- "balance_final"      

# Order and sort
portfolio <- portfolio[c("teams","aid","uid","balance_final")]
portfolio <- portfolio[order(portfolio$teams,portfolio$aid),]

# Save
save(portfolio,
     file = paste(comb, "portfolio.RData", sep='/')) 
   
      
# Clean-Up Data Position Data Table #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
position$type <- NULL

# Rename Variables
names(position)[names(position) == "position_id"] <- "pid"
names(position)[names(position) == "portfolio_id"] <- "aid"     

# Hypothesis ID  
position$hid <- position %>% group_indices(asset_id) 
position$hid <- position$hid - position$teams * 9
position$asset_id <- NULL       

# Order and Sort
position <- position[c("teams","pid","aid","hid","shares")]
position <- position[order(position$teams,position$pid),]

# Save
save(position,
     file = paste(comb, "position.RData", sep='/'))      
        

# Clean-Up Data Transactions Data Table #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
transaction$type <- NULL

# Rename Variables
names(transaction)[names(transaction) == "transaction_id"] <- "tid"
names(transaction)[names(transaction) == "position_id"] <- "pid"
names(transaction)[names(transaction) == "cost"] <- "investment" 
names(transaction)[names(transaction) == "old_price"] <- "p_initial"  
names(transaction)[names(transaction) == "new_price"] <- "p_new" 
names(transaction)[names(transaction) == "date"] <- "timestamp"

# Order and Sort
transaction <- transaction[c("teams","timestamp","tid","pid","balance","investment","p_initial","p_new","shares")]
transaction <- transaction[order(transaction$teams,transaction$timestamp),]

# Save
save(transaction,
     file = paste(comb, "transaction.RData", sep='/'))     
  

# Clean-Up Data Results Data Table #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #  

# Reshape
t1 <- reshape(TeamResults, direction='long', 
        varying=c('hypo1_decision',	
                  'hypo2_decision',	
                  'hypo3_decision',	
                  'hypo4_decision',	
                  'hypo5_decision',	
                  'hypo6_decision',	
                  'hypo7_decision',	
                  'hypo8_decision',	
                  'hypo9_decision'),
        timevar='hid',
        times=c('1', '2','3','4','5','6','7','8','9'),
        v.names='decision',
        idvar='teamresults_id')  

t2 <- reshape(TeamResults, direction='long', 
        varying=c('hypo1_confidence',	
                 'hypo2_confidence',	
                 'hypo3_confidence',	
                 'hypo4_confidence',	
                 'hypo5_confidence',	
                 'hypo6_confidence',	
                 'hypo7_confidence',	
                 'hypo8_confidence',	
                 'hypo9_confidence'),
        timevar='hid',
        times=c('1', '2','3','4','5','6','7','8','9'),
        v.names='confidence',
        idvar=c('teamresults_id'))  
  
t3 <- reshape(TeamResults, direction='long', 
        varying=c('hypo1_similarity',	
                  'hypo2_similarity',	
                  'hypo3_similarity',	
                  'hypo4_similarity',	
                  'hypo5_similarity',	
                  'hypo6_similarity',	
                  'hypo7_similarity',	
                  'hypo8_similarity',	
                  'hypo9_similarity'),
        timevar='hid',
        times=c('1', '2','3','4','5','6','7','8','9'),
        v.names='similarity',
        idvar=c('teamresults_id'))  

t1 <- t1[c("teamresults_id",
           "teams",
           "hid",
           "decision")]

TR <- merge(t1,t2[,c("teamresults_id","hid","confidence")], by=c("teamresults_id","hid"))
TeamResults <- merge(TR,t3[,c("teamresults_id","hid","similarity")], by=c("teamresults_id","hid"))

# Rename Variables
names(TeamResults)[names(TeamResults) == "teamresults_id"] <- "uid" 
TeamResults$decision <- ifelse(TeamResults$decision=="Yes",1,0)

# Save=
save(TeamResults,
     file = paste(proc, "TeamResults.RData", sep='/'))    
     
 
        
### Fundamental Values                                                         ###
### -------------------------------------------------------------------------- ### 
n <- 70
hid<- 1:9

# Set Fundamentals
fundamental <- data.frame(hid)

fundamental$fv <- ifelse(fundamental$hid==1,26/n,".")
fundamental$fv <- ifelse(fundamental$hid==2,15/n,fundamental$fv)
fundamental$fv <- ifelse(fundamental$hid==3,16/n,fundamental$fv)
fundamental$fv <- ifelse(fundamental$hid==4,23/n,fundamental$fv)
fundamental$fv <- ifelse(fundamental$hid==5,59/n,fundamental$fv)
fundamental$fv <- ifelse(fundamental$hid==6,23/n,fundamental$fv)
fundamental$fv <- ifelse(fundamental$hid==7,4/n,fundamental$fv)
fundamental$fv <- ifelse(fundamental$hid==8,4/n,fundamental$fv)
fundamental$fv <- ifelse(fundamental$hid==9,4/n,fundamental$fv)

# Determine Confidence Intervals
fundamental$fv <- as.numeric(fundamental$fv)

fundamental$fv_95l <- fundamental$fv - ((fundamental$fv * (1 - fundamental$fv))/n)^0.5 * qnorm(0.975)
fundamental$fv_95u <- fundamental$fv + ((fundamental$fv * (1 - fundamental$fv))/n)^0.5 * qnorm(0.975)      

# Save
save(fundamental,
     file = paste(proc, "Fundamentals.RData", sep='/'))    
     


### Merge Datasets                                                             ###
### -------------------------------------------------------------------------- ###
        
# Asset Holdings
Holdings<- merge(position,portfolio[,c("aid","uid")],by="aid")
Holdings<- merge(Holdings,asset[,c("hid","teams","hypothesis")],by=c("hid","teams"))

Holdings$aid <- NULL
Holdings$pid <- NULL

Holdings <- Holdings[c("teams", "uid", "hid", "hypothesis","shares")]
Holdings <- Holdings[order(Holdings$teams,Holdings$uid,Holdings$hid),]

setwd(proc)
save(Holdings, file="Holdings.RData")

# Prices
Prices <- asset
Prices$long_outstanding <- NULL
Prices$short_outstanding <- NULL

save(Prices,
     file=paste(proc, "Prices.RData", sep='/'))


# Transactions
Transactions<- merge(transaction,position[,c("pid","aid","hid")],by="pid")
Transactions<- merge(Transactions,portfolio[,c("aid","uid")],by="aid")
Transactions<- merge(Transactions,asset[,c("teams","hypothesis","hid")],by=c("teams","hid")) 

Transactions$pid <- NULL
Transactions$aid <- NULL

Transactions <- Transactions[c("teams","hid","hypothesis","timestamp","uid","tid","balance","investment","p_initial","p_new","shares")]
Transactions <- Transactions[order(Transactions$teams,Transactions$hid,Transactions$timestamp),]

save(Transactions,
     file = paste(proc, "Transactions.RData", sep='/'))
        
        
### Hourly Balanced Panel                                                      ###
### -------------------------------------------------------------------------- ###
        
BalancedPanel <- Transactions

BalancedPanel$datetime <- as.POSIXct(BalancedPanel$timestamp)
BalancedPanel$day <- mday(BalancedPanel$datetime)
BalancedPanel$hour <- hour(BalancedPanel$datetime)

BalancedPanel <- aggregate(BalancedPanel$p_new, 
                           by = list(
                             BalancedPanel$teams,
                             BalancedPanel$hid,
                             BalancedPanel$day, 
                             BalancedPanel$hour ), 
                           FUN = "mean")        

names(BalancedPanel)[names(BalancedPanel) == "Group.1"] <- "teams"
names(BalancedPanel)[names(BalancedPanel) == "Group.2"] <- "hid"
names(BalancedPanel)[names(BalancedPanel) == "Group.3"] <- "day"
names(BalancedPanel)[names(BalancedPanel) == "Group.4"] <- "hour"
names(BalancedPanel)[names(BalancedPanel) == "x"] <- "price"

BalancedPanel<-complete(BalancedPanel, day, hour, teams, hid)

BalancedPanel<-BalancedPanel[!(BalancedPanel$day == 2  & BalancedPanel$hour<16),]
BalancedPanel<-BalancedPanel[!(BalancedPanel$day == 12 & BalancedPanel$hour>15),]  

BalancedPanel<-merge(BalancedPanel,fundamental[,c("hid","fv")],by=c("hid"))
BalancedPanel$time <- BalancedPanel %>% group_indices(day, hour) 

BalancedPanel$day <- NULL
BalancedPanel$hour<- NULL

BalancedPanel <- BalancedPanel[c("teams", "hid", "time","price","fv")]
BalancedPanel <- BalancedPanel[order(BalancedPanel$teams,BalancedPanel$hid,BalancedPanel$time),]

BalancedPanel <- BalancedPanel %>% fill(price)

BalancedPanel$ae <- abs(BalancedPanel$price - BalancedPanel$fv)

# Save
save(BalancedPanel, 
     file = paste(proc, "BalancedPanel.RData", sep='/'))

