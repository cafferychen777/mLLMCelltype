library('variancePartition')
library('edgeR')
library('BiocParallel')
library('plyr')

# path info
input_file_prefix <- "../results/covariate_modeling/input/"
output_file_prefix <- "../results/covariate_modeling/output/"
# read cts from input file folder
cts <- list.files(input_file_prefix)
# for testing:
# cts = c("EC_arterial")
# end of testing line
for (celltype in cts){
  print(celltype)
  # read in data:
  count_matrix_path = paste(input_file_prefix, celltype, "/sample_gene_sums.csv", sep="")
  if (file.exists(count_matrix_path)) {
    output_dir = paste(output_file_prefix, celltype, sep="")
    if (!file.exists(paste(output_dir, "/mm_output.tsv", sep=""))){
      print('Continuing to modeling...')
      countMatrix <- as.matrix(t(read.csv(count_matrix_path,header=TRUE,row.names = 1)))
      metadata_path = paste(input_file_prefix, celltype, "/sample_design_matrix.csv", sep="")
      metadata <- read.csv(metadata_path, header=TRUE, row.names=1)
      names(metadata)[names(metadata) == 'sample.1'] <- 'sample' # LISA: test line
      if (dim(metadata)[1] < 5){
        print("less than 5 samples detected. skipping this celltype")
        next
      }
      # re-order factor levels, so that most prevalent comes first. This should make the modeling
      # more robust
      metadata$sex <- as.factor(metadata$sex) # ensure it is a factor
      metadata$sex <- relevel(metadata$sex, ref="male")
      if ("ethnicity" %in% names(metadata)){
        metadata$ethnicity <- as.factor(metadata$ethnicity)
        metadata$ethnicity <- relevel(metadata$ethnicity, ref="white")
      }
      # set non-smoker to reference, for interpretability
      if ("smoking_status" %in% names(metadata)){
        metadata$smoking_status <- as.factor(metadata$smoking_status)
        metadata$smoking_status <- relevel(metadata$smoking_status, ref="never")
      }
      # Standard usage of limma/voom
      geneExpr = DGEList( countMatrix ) #[isexpr,] ) # creates object in certain format, i.e. a counts object
      # ($counts), that has the count matrix, and a samples object ($samples) that has samples as rows,
      # and for each sample: group assignments, lib.sizes, and norm factors
      geneExpr = calcNormFactors( geneExpr ) # this does a SCRAN-like normalization, normalizing such that
      # most genes have fold changes (between samples) as close to 1 as possible.

      # make this vignette faster by analyzing a subset of genes, e.g. when doing a test-run:
      # geneExpr = geneExpr[1:100,]

      # DREAM ANALYSIS

      # Specify parallel processing parameters
      # this is used implicitly by dream() to run in parallel
      param = SnowParam(progressbar=TRUE)
      register(param)

      # check if there are subjects with multiple samples. If so, consider using subject as a random effect.
      # We did not do that in this script.
      print("Number of subjects equal to number of samples?")
      print(length(unique(metadata$subject_ID)) == dim(metadata)[1])

      # The variable to be tested must be a fixed effect
      if ("anatomical_region_ccf_score" %in% names(metadata)){
        form <- "~ sex + age + smoking_status_num + BMI + ethnicity + anatomical_region_ccf_score + nose + (1|dataset)" # (1|subject_ID)"
      } else {
        form <- "~ sex + age + smoking_status_num + BMI + ethnicity + (1|dataset)" # + (1|subject_ID)"
      }
      if (!"ethnicity" %in% names(metadata)){
        form = sub(" + ethnicity", "", form, fixed=TRUE)
      }
      if (!"smoking_status_num" %in% names(metadata)){
        form = sub(" + smoking_status_num", "", form, fixed=TRUE)
      }
      else if (nlevels(metadata$ethnicity) == 1){
        print("dropping ethnicity, since it has only one category for this celltype.")
        form = sub(" + ethnicity", "", form, fixed=TRUE)
      }
      if (nlevels(metadata$sex) == 1){
        print("dropping sex, since it has only one category for this celltype.")
        form = sub(" sex +", "", form, fixed=TRUE)
      }
      if (!"nose" %in% names(metadata)){
        form = sub(" + nose", "", form, fixed=TRUE)
      }
      else if (length(unique(metadata$nose)) == 1){
        form = sub(" + nose", "", form, fixed=TRUE)
      }
      # estimate weights using linear mixed model of dream
      vobjDream = voomWithDreamWeights(geneExpr, form, metadata)
      # Fit the dream model on each gene
      # By default, uses the Satterthwaite approximation for the hypothesis test
      fitmm = dream(vobjDream, form, metadata )
      # store results:
      if (!dir.exists(output_file_prefix)){
        dir.create(output_file_prefix)
      }
      if (!dir.exists(output_dir)){
        dir.create(output_dir)
      }
      write.fit(fitmm, file=paste(output_dir, "/mm_output.tsv", sep=""))
      write.csv(geneExpr$samples,file=paste(output_dir, "/mm_norm_factors.tsv", sep=""))
    }
  }
}
