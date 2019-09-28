library(checkpoint)
checkpointDir <- '/checkpoint'
dir.create(checkpointDir)
dir.create(file.path(checkpointDir, ".checkpoint"))
checkpoint("2019-08-13", checkpointLocation = checkpointDir)

