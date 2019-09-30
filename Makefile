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

# get list of R packages needed by checkpoint
get-R-packages:
	cat */*.Rmd */*.R | grep library >| R_libraries.R

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

