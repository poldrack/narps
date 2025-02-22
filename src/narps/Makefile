# Makefile for NARPS analyses

# variables
# use NARPS docker image from poldrack dockerhub
# set to your username if you wish to push custom version to a different dockerhub acct
DOCKER_USERNAME = poldrack

# need this for simulations
NARPS_BASEDIR_SIMULATED = ${NARPS_BASEDIR}_simulated

# code to check environment variables
# from https://stackoverflow.com/questions/4728810/makefile-variable-as-prerequisite

guard-%:
	@ if [ "${${*}}" = "" ]; then \
		echo "Environment variable $* not set"; \
		exit 1; \
	fi

# from https://stackoverflow.com/questions/18136918/how-to-get-current-relative-directory-of-your-makefile

current_dir = $(shell pwd)

# testing functions

test:
	pytest -q tests.py

check-style:
	flake8 --show-source  | tee stylecheck.out

# get list of R packages needed by checkpoint
get-R-packages:
	cat *.Rmd | grep library >| R_libraries.R


# commands to run analyses via docker

run-all: run-PrepareMaps run-PrepareMetadata run-AnalyzeMaps run-AnalyzeDecisions run-ConsensusAnalysis run-ALE

run-PrepareMaps: guard-DOCKER_USERNAME guard-NARPS_BASEDIR guard-DATA_URL
	docker run -e "DATA_URL=$(DATA_URL)" -v $(current_dir):/analysis -v $(NARPS_BASEDIR):/data $(DOCKER_USERNAME)/narps-analysis /bin/bash -c "source /etc/fsl/5.0/fsl.sh;python /analysis/PreprocessMaps.py"

run-PrepareMetadata: guard-DOCKER_USERNAME guard-NARPS_BASEDIR 
	docker run -v $(current_dir):/analysis -v $(NARPS_BASEDIR):/data $(DOCKER_USERNAME)/narps-analysis /bin/bash -c "source /etc/fsl/5.0/fsl.sh;python /analysis/PrepareMetadata.py"

run-AnalyzeMaps: guard-DOCKER_USERNAME guard-NARPS_BASEDIR 
	docker run -v $(current_dir):/analysis -v $(NARPS_BASEDIR):/data $(DOCKER_USERNAME)/narps-analysis /bin/bash -c "source /etc/fsl/5.0/fsl.sh;python /analysis/AnalyzeMaps.py" 

run-ConsensusAnalysis: guard-DOCKER_USERNAME guard-NARPS_BASEDIR 
	docker run -v $(current_dir):/analysis -v $(NARPS_BASEDIR):/data $(DOCKER_USERNAME)/narps-analysis /bin/bash -c "source /etc/fsl/5.0/fsl.sh;python /analysis/ConsensusAnalysis.py"
	docker run -v $(current_dir):/analysis -v $(NARPS_BASEDIR):/data $(DOCKER_USERNAME)/narps-analysis /bin/bash -c "source /etc/fsl/5.0/fsl.sh;python /analysis/ClusterImageCorrelation.py"

run-ALE: guard-DOCKER_USERNAME guard-NARPS_BASEDIR 
	docker run -v $(current_dir):/analysis -v $(NARPS_BASEDIR):/data $(DOCKER_USERNAME)/narps-analysis /bin/bash -c "source /etc/fsl/5.0/fsl.sh;python /analysis/MetaAnalysis.py"

run-AnalyzeDecisions: guard-DOCKER_USERNAME guard-NARPS_BASEDIR get-R-packages
	docker run -v $(current_dir):/analysis -v $(NARPS_BASEDIR):/data $(DOCKER_USERNAME)/narps-analysis Rscript -e 'library(knitr);library(rmarkdown); rmarkdown::render("/analysis/DecisionAnalysis.Rmd", "html_document", output_dir = "/data/figures")'

run-MakeFigures: guard-DOCKER_USERNAME guard-NARPS_BASEDIR
	docker run -v $(current_dir):/analysis -v $(NARPS_BASEDIR):/data $(DOCKER_USERNAME)/narps-analysis /bin/bash -c "source /etc/fsl/5.0/fsl.sh;python /analysis/MakeSupplementaryFigure1.py"

# commands to run simulated data analyses via docker

run-all-simulated: run-PrepareMaps-simulated run-PrepareMetadata-simulated run-AnalyzeMaps-simulated run-AnalyzeDecisions-simulated run-ConsensusAnalysis-simulated

run-PrepareMaps-simulated: guard-DOCKER_USERNAME guard-NARPS_BASEDIR guard-DATA_URL
	-mkdir $(NARPS_BASEDIR_SIMULATED)
	docker run -e "DATA_URL=$(DATA_URL)" -v $(current_dir):/analysis -v $(NARPS_BASEDIR):/data -v $(NARPS_BASEDIR_SIMULATED):/data_simulated $(DOCKER_USERNAME)/narps-analysis /bin/bash -c "source /etc/fsl/5.0/fsl.sh;python /analysis/PreprocessMaps.py -s"

run-PrepareMetadata-simulated: guard-DOCKER_USERNAME guard-NARPS_BASEDIR 
	docker run -v $(current_dir):/analysis -v $(NARPS_BASEDIR_SIMULATED):/data $(DOCKER_USERNAME)/narps-analysis /bin/bash -c "source /etc/fsl/5.0/fsl.sh;python /analysis/PrepareMetadata.py"

# need to link both directories to the basedir
run-AnalyzeMaps-simulated: guard-DOCKER_USERNAME guard-NARPS_BASEDIR 
	docker run -v $(current_dir):/analysis -v $(NARPS_BASEDIR_SIMULATED):/data_simulated -v $(NARPS_BASEDIR_SIMULATED):/data $(DOCKER_USERNAME)/narps-analysis /bin/bash -c "source /etc/fsl/5.0/fsl.sh;python /analysis/AnalyzeMaps.py"

run-ConsensusAnalysis-simulated: guard-DOCKER_USERNAME guard-NARPS_BASEDIR 
	docker run -v $(current_dir):/analysis -v $(NARPS_BASEDIR_SIMULATED):/data -v $(NARPS_BASEDIR_SIMULATED):/data_simulated $(DOCKER_USERNAME)/narps-analysis /bin/bash -c "source /etc/fsl/5.0/fsl.sh;python /analysis/ConsensusAnalysis.py"
	docker run -v $(current_dir):/analysis -v $(NARPS_BASEDIR):/data -v $(NARPS_BASEDIR_SIMULATED):/data_simulated $(DOCKER_USERNAME)/narps-analysis /bin/bash -c "source /etc/fsl/5.0/fsl.sh;python /analysis/CheckSimulatedValues.py"


run-AnalyzeDecisions-simulated: guard-DOCKER_USERNAME guard-NARPS_BASEDIR get-R-packages
	docker run -v $(current_dir):/analysis -v $(NARPS_BASEDIR_SIMULATED):/data $(DOCKER_USERNAME)/narps-analysis Rscript -e 'library(knitr);library(rmarkdown); rmarkdown::render("/analysis/DecisionAnalysis.Rmd", "html_document", output_dir = "/data/figures")'

run-MakeFigures-simulated: guard-DOCKER_USERNAME guard-NARPS_BASEDIR
	docker run -v $(current_dir):/analysis -v $(NARPS_BASEDIR_SIMULATED):/data $(DOCKER_USERNAME)/narps-analysis /bin/bash -c "source /etc/fsl/5.0/fsl.sh;python /analysis/MakeSupplementaryFigure1.py"



# commands for building and testing docker image

docker-deploy: docker-login docker-upload

docker-login: guard-DOCKER_USERNAME guard-DOCKER_PASSWORD
	docker login --username=$(DOCKER_USERNAME) --password=$(DOCKER_PASSWORD)

docker-upload: guard-DOCKER_USERNAME
	docker push $(DOCKER_USERNAME)/narps-analysis

docker-build: guard-DOCKER_USERNAME
	docker build -t $(DOCKER_USERNAME)/narps-analysis .

# add -p 8888:8888 for jupyter
shell: guard-DOCKER_USERNAME guard-NARPS_BASEDIR guard-DATA_URL
	docker run -e "DATA_URL=$(DATA_URL)" -it --entrypoint=bash -v $(current_dir):/analysis -v $(NARPS_BASEDIR):/data $(DOCKER_USERNAME)/narps-analysis 

# commands to run on local machine

install-R-packages:
	echo 'install.packages(c("tidyverse","lme4","lmerTest","emmeans","pscl","arm","MuMIn"), repos="http://cran.us.r-project.org", dependencies=TRUE)' > /tmp/packages.R && Rscript /tmp/packages.R


run-all-local: run-PrepareMaps-local run-PrepareMetadata-local run-AnalyzeMaps-local run-ConsensusAnalysis-local # run-AnalyzeDecisions-local

run-PrepareMaps-local:
	python PreprocessMaps.py

run-PrepareMetadata-local:
	python PrepareMetadata.py

run-AnalyzeMaps-local:
	python AnalyzeMaps.py
	python ClusterImageCorrelation.py

run-ConsensusAnalysis-local:
	python ConsensusAnalysis.py

run-AnalyzeDecisions-local: get-R-packages
	 Rscript -e 'library(knitr);library(rmarkdown); rmarkdown::render("DecisionAnalysis.Rmd", "html_document", output_dir = "$(NARPS_BASEDIR)/figures")'

