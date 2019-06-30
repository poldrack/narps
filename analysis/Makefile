# code to check environment variables
# from https://stackoverflow.com/questions/4728810/makefile-variable-as-prerequisite

guard-%:
	@ if [ "${${*}}" = "" ]; then \
		echo "Environment variable $* not set"; \
		exit 1; \
	fi

# from https://stackoverflow.com/questions/18136918/how-to-get-current-relative-directory-of-your-makefile
current_dir = $(shell pwd)
# use NARPS docker image from poldrack dockerhub
# set to your username if you wish to push custom version to a different dockerhub acct
DOCKER_USERNAME = poldrack

test:
	pytest -q narps.py
run:
	python narps.py

docker-deploy: docker-login docker-upload

docker-login: guard-DOCKER_USERNAME guard-DOCKER_PASSWORD
	docker login --username=$(DOCKER_USERNAME) --password=$(DOCKER_PASSWORD)

docker-upload: guard-DOCKER_USERNAME
	docker push $(DOCKER_USERNAME)/narps-analysis

docker-build: guard-DOCKER_USERNAME
	docker build -t $(DOCKER_USERNAME)/narps-analysis .

shell: guard-DOCKER_USERNAME guard-NARPS_BASEDIR 
	docker run -it --entrypoint=bash -v $(current_dir):/analysis -v $(NARPS_BASEDIR):/data -p 8888:8888 $(DOCKER_USERNAME)/narps-analysis 


# this has same functionality as running the PrepareMaps function below, except
# that it doesn't produce reports
run-narps: guard-DOCKER_USERNAME guard-NARPS_BASEDIR 
	docker run -v $(current_dir):/analysis -v $(NARPS_BASEDIR):/data $(DOCKER_USERNAME)/narps-analysis python /analysis/narps.py

run-all: run-PrepareMaps run-PrepareMetadata run-AnalyzeMaps run-AnalyzeDecisions run-ConsensusAnalysis

run-PrepareMetadata: guard-DOCKER_USERNAME guard-NARPS_BASEDIR 
	docker run -v $(current_dir):/analysis -v $(NARPS_BASEDIR):/data $(DOCKER_USERNAME)/narps-analysis jupyter nbconvert --to notebook --inplace --execute /analysis/PrepareMetadata.ipynb

# need to source fsl setup to use smoothest
run-PrepareMaps: guard-DOCKER_USERNAME guard-NARPS_BASEDIR 
	docker run -v $(current_dir):/analysis -v $(NARPS_BASEDIR):/data $(DOCKER_USERNAME)/narps-analysis /bin/bash -c "source /etc/fsl/5.0/fsl.sh;jupyter nbconvert --to notebook --inplace --ExecutePreprocessor.timeout=-1 --execute /analysis/PrepareMaps.ipynb"

# requires PrepareMetadata and PrepareMaps to have been run
run-AnalyzeMaps: guard-DOCKER_USERNAME guard-NARPS_BASEDIR 
	docker run -v $(current_dir):/analysis -v $(NARPS_BASEDIR):/data $(DOCKER_USERNAME)/narps-analysis jupyter nbconvert --to notebook --inplace --ExecutePreprocessor.timeout=-1 --execute /analysis/AnalyzeMaps.ipynb

# requires PrepareMetadata and PrepareMaps to have been run
run-ConsensusAnalysis: guard-DOCKER_USERNAME guard-NARPS_BASEDIR 
	docker run -v $(current_dir):/analysis -v $(NARPS_BASEDIR):/data $(DOCKER_USERNAME)/narps-analysis jupyter nbconvert --to notebook --inplace --ExecutePreprocessor.timeout=-1 --execute /analysis/ConsensusAnalysis.ipynb

# requires AnalyzeMaps to have been run, for pattern distance metadata
run-AnalyzeDecisions: guard-DOCKER_USERNAME guard-NARPS_BASEDIR 
	docker run -v $(current_dir):/analysis -v $(NARPS_BASEDIR):/data $(DOCKER_USERNAME)/narps-analysis Rscript -e 'library(rmarkdown); rmarkdown::render("/analysis/DecisionAnalysis.Rmd", "html_document")'