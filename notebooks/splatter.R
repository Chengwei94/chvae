library(ExperimentHub)
library(splatter)
library(scater)
library(scran)
library(Matrix)
library(Seurat)
library(SingleCellExperiment)
library(DropletUtils)

eh = ExperimentHub()
query(eh, "Kang")
sce <- eh[["EH2259"]]
clusters <- quickCluster(sce)
# sce <- computeSumFactors(sce, clusters)
sce <- logNormCounts(sce)
tmp <- modelGeneVar(sce)
hvgs <- getTopHVGs(tmp, n=2000)
sce <- sce[hvgs]


simulations <- list()
de_prob <- c(0, 0, 0, 0.2, 0.5, 1)
de_facloc <- c(0, 0, 0, 0.2, 0.5, 1)
i <- 1
for (celltype in levels(sce$cell)[c(1:6)]){
  counts <- subset(sce, , cell==celltype)
  counts <- as.matrix(assay(counts , "counts"))
  if(dim(counts)[2] > 600){
    counts <- counts[, 1:600]
  }
  prop <- runif(1, min = 0.4, max = 0.6)
  # Print the random number
  params <- splatEstimate(counts)
  params_modified <- setParams(params, 
                               batchCells  = 500,
                               group.prob = c(prop, 1 - prop),
                               de.prob = de_prob[i],
                               de.facLoc = de_facloc[i],
                               dropout.type = "none"
  )    # Log-fold change location for DE genes
  sim_data_modified <- splatSimulate(params_modified, method = "groups")
  simulations[[celltype]] <- sim_data_modified
  i <- i + 1
  # print(sim_data_modified)
  # sim_data_modified <- logNormCounts(sim_data_modified)
  # sim_data_modified <- runPCA(sim_data_modified)
  # sim_data_modified <- runUMAP(sim_data_modified)
  # plotPCA(sim_data_modified, colour_by = "Group")
  # plotUMAP(sim_data_modified, colour_by = "Group")
}

simulated_counts <- list()
metadata <- list()

for (celltype in names(simulations)){
  simulated_counts[[celltype]] <- (assay(simulations[[celltype]], "counts"))
  metadata[[celltype]] <- as.data.frame(colData(simulations[[celltype]]))
  rownames(metadata[[celltype]]) <- paste0(celltype, rownames(metadata[[celltype]]))
  metadata[[celltype]]$celltype <- celltype
}

mat <- do.call("cbind", simulated_counts)
metadata <- do.call("rbind", metadata)

write10xCounts('simulation_data/simulation1', mat, overwrite = TRUE)
write.csv(metadata, "simulation_data/simulation1_meta.csv")
