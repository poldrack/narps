# Makefile for NARPS analyses

# variables
# use NARPS docker image from poldrack dockerhub
# set to your username if you wish to push custom version to a different dockerhub acct
DOCKER_USERNAME = poldrack

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
	pytest -q narps.py

check-style:
	flake8 --show-source  >| stylecheck.out


# commands to run analyses via docker

run-all: run-PrepareMaps run-PrepareMetadata run-AnalyzeMaps run-AnalyzeDecisions run-ConsensusAnalysis

run-PrepareMaps: guard-DOCKER_USERNAME guard-NARPS_BASEDIR guard-DATA_URL
	docker run -e "DATA_URL=$(DATA_URL)" -v $(current_dir):/analysis -v $(NARPS_BASEDIR):/data $(DOCKER_USERNAME)/narps-analysis /bin/bash -c "source /etc/fsl/5.0/fsl.sh;python /analysis/narps.py"

run-PrepareMetadata: guard-DOCKER_USERNAME guard-NARPS_BASEDIR 
	docker run -v $(current_dir):/analysis -v $(NARPS_BASEDIR):/data $(DOCKER_USERNAME)/narps-analysis python /analysis/PrepareMetadata.py

run-AnalyzeMaps: guard-DOCKER_USERNAME guard-NARPS_BASEDIR 
	docker run -v $(current_dir):/analysis -v $(NARPS_BASEDIR):/data $(DOCKER_USERNAME)/narps-analysis python /analysis/AnalyzeMaps.py

run-ConsensusAnalysis: guard-DOCKER_USERNAME guard-NARPS_BASEDIR 
	docker run -v $(current_dir):/analysis -v $(NARPS_BASEDIR):/data $(DOCKER_USERNAME)/narps-analysis python /analysis/ConsensusAnalysis.py

run-AnalyzeDecisions: guard-DOCKER_USERNAME guard-NARPS_BASEDIR 
	docker run -v $(current_dir):/analysis -v $(NARPS_BASEDIR):/data $(DOCKER_USERNAME)/narps-analysis Rscript -e 'library(rmarkdown); rmarkdown::render("/analysis/DecisionAnalysis.Rmd", "html_document", output_dir = "/data")'


# commands for building and testing docker image

docker-deploy: docker-login docker-upload

docker-login: guard-DOCKER_USERNAME guard-DOCKER_PASSWORD
	docker login --username=$(DOCKER_USERNAME) --password=$(DOCKER_PASSWORD)

docker-upload: guard-DOCKER_USERNAME
	docker push $(DOCKER_USERNAME)/narps-analysis

docker-build: guard-DOCKER_USERNAME
	docker build -t $(DOCKER_USERNAME)/narps-analysis .

shell: guard-DOCKER_USERNAME guard-NARPS_BASEDIR 
	docker run -it --entrypoint=bash -v $(current_dir):/analysis -v $(NARPS_BASEDIR):/data -p 8888:8888 $(DOCKER_USERNAME)/narps-analysis 

# commands to run on local machine

install-R-packages:
	echo 'install.packages(c("tidyverse","lme4","lmerTest","emmeans","pscl","arm","MuMIn"), repos="http://cran.us.r-project.org", dependencies=TRUE)' > /tmp/packages.R && Rscript /tmp/packages.R

run-all-local: run-PrepareMaps-local run-PrepareMetadata-local run-AnalyzeMaps-local run-ConsensusAnalysis-local # run-AnalyzeDecisions-local

run-PrepareMaps-local:
	python narps.py

run-PrepareMetadata-local:
	python PrepareMetadata.py

run-AnalyzeMaps-local:
	python AnalyzeMaps.py

run-ConsensusAnalysis-local:
	python ConsensusAnalysis.py

run-AnalyzeDecisions-local:
	 Rscript -e 'library(rmarkdown); rmarkdown::render("DecisionAnalysis.Rmd", "html_document")'

