test:
	pytest -q narps.py
run:
	python narps.py

docker-deploy: docker-login docker-upload

docker-login:
	docker login --username=$(DOCKER_USERNAME) --password=$(DOCKER_PASSWORD)

docker-upload:
	docker push $(DOCKER_USERNAME)/narps-analysis

docker-build:
	docker build -t $(DOCKER_USERNAME)/narps-analysis .

shell:
	docker run -it --entrypoint=bash -v /Users/poldrack/Dropbox/code/narps/analysis:/analysis -v /Users/poldrack/data_unsynced/NARPS:/data -p 8888:8888 $(DOCKER_USERNAME)/narps-analysis 


# this has same functionality as running the PrepareMaps function below, except
# that it doesn't produce reports
run-narps:
	docker run -v /Users/poldrack/Dropbox/code/narps/analysis:/analysis -v /Users/poldrack/data_unsynced/NARPS:/data $(DOCKER_USERNAME)/narps-analysis python /analysis/narps.py

run-all: run-PrepareMaps run-PrepareMetadata run-AnalyzeMaps run-AnalyzeDecisions run-ConsensusAnalysis

run-PrepareMetadata:
	docker run -v /Users/poldrack/Dropbox/code/narps/analysis:/analysis -v /Users/poldrack/data_unsynced/NARPS:/data $(DOCKER_USERNAME)/narps-analysis jupyter nbconvert --to notebook --inplace --execute /analysis/PrepareMetadata.ipynb

# need to source fsl setup to use smoothest
run-PrepareMaps:
	docker run -v /Users/poldrack/Dropbox/code/narps/analysis:/analysis -v /Users/poldrack/data_unsynced/NARPS:/data $(DOCKER_USERNAME)/narps-analysis /bin/bash -c "source /etc/fsl/5.0/fsl.sh;jupyter nbconvert --to notebook --inplace --ExecutePreprocessor.timeout=-1 --execute /analysis/PrepareMaps.ipynb"

# requires PrepareMetadata and PrepareMaps to have been run
run-AnalyzeMaps:
	docker run -v /Users/poldrack/Dropbox/code/narps/analysis:/analysis -v /Users/poldrack/data_unsynced/NARPS:/data $(DOCKER_USERNAME)/narps-analysis jupyter nbconvert --to notebook --inplace --ExecutePreprocessor.timeout=-1 --execute /analysis/AnalyzeMaps.ipynb

# requires PrepareMetadata and PrepareMaps to have been run
run-ConsensusAnalysis:
	docker run -v /Users/poldrack/Dropbox/code/narps/analysis:/analysis -v /Users/poldrack/data_unsynced/NARPS:/data $(DOCKER_USERNAME)/narps-analysis jupyter nbconvert --to notebook --inplace --ExecutePreprocessor.timeout=-1 --execute /analysis/ConsensusAnalysis.ipynb

# requires AnalyzeMaps to have been run, for pattern distance metadata
run-AnalyzeDecisions:
	docker run -v /Users/poldrack/Dropbox/code/narps/analysis:/analysis -v /Users/poldrack/data_unsynced/NARPS:/data $(DOCKER_USERNAME)/narps-analysis Rscript -e 'library(rmarkdown); rmarkdown::render("/analysis/DecisionAnalysis.Rmd", "html_document")'