suppressMessages(library(data.table))
suppressMessages(library(readxl))
suppressMessages(library(Seurat))
suppressMessages(library(ggplot2))
suppressMessages(library(ggpubr))
suppressMessages(library(Nebulosa))
suppressMessages(library(ggrastr))
suppressMessages(library(cowplot))
suppressMessages(library(reshape2))
suppressMessages(library(RColorBrewer))
suppressMessages(library(pheatmap))
suppressMessages(library(dplyr))
suppressMessages(library(viridis))

setwd('data2024.v2/')

rename2 <- fread(paste0('anno.264clusters.renameV1212.txt'), header = TRUE, sep = "\t", data.table = F)
rownames(rename2) <- rename2$anno2
mat_columnanno <- read.delim('data2024/column2annot_man_res_corrected.csv', sep = ',', header = T)

get_celldense <- function(sampletag, min_d = 5) {
    mat_cl <- fread(paste0('cell-meta-matrix_1105/cell-meta-matrix.', sampletag, '.tsv'), header = TRUE, sep = "\t", data.table = F)

    ### rename class/subclass/cluster
    mat_cl$class <- lapply(rename2[mat_cl$cluster, 'FullName'], function(item) strsplit(item, split = '[ ]')[[1]][1])
    mat_cl$subclass <- lapply(rename2[mat_cl$cluster, 'FullName'], function(item) strsplit(strsplit(item, split = '[ ]')[[1]][2], split = '[.]')[[1]][1])
    mat_cl$celltype <- rename2[mat_cl$cluster, 'FullName']

    ### column
    mat_column <- fread(paste0('data2024/cell_column/', sampletag, '_cell2column.csv'), header = TRUE, sep = ",", data.table = F)
    mat_cl <- merge(mat_cl, mat_column, by = 'cell_id')
    print(dim(mat_cl))

    colnames(mat_cl)[colnames(mat_cl) == 'column_id'] <- 'column_id_ori'
    a <- data.frame(row.names = sort(unique(mat_cl$column_id_ori)), reid = 1:length(unique(mat_cl$column_id_ori)))
    mat_cl$column_id <- a[as.character(mat_cl$column_id_ori), 'reid']

    ### seurat object
    sob <- readRDS(paste0('spa_matrix/spa_matrix.', sampletag, '.rds'))
    sob <- subset(sob, cells = as.character(mat_cl$cell_id))
    sob@meta.data <- mat_cl
    rownames(sob@meta.data) <- mat_cl$cell_id
    table(rownames(sob@meta.data) == colnames(sob))
    table(rownames(sob@meta.data) == mat_cl$cell_id)
    
    ### read in the column annotation
    mat_columnanno <- read.delim('data2024/column2annot_man_res_corrected.csv', sep = ',', header = T)
    mat_columnanno <- mat_columnanno[mat_columnanno$section == sampletag,]
    mat_columnanno <- mat_columnanno[order(mat_columnanno$column),]

    ###
    sob@meta.data$gs <- NA
    sob@meta.data$area <- NA
    for (i in 1:nrow(mat_columnanno)) {
        if (mat_columnanno[i,'column'] %in% mat_cl$column_id_ori) {
            column_mid = mat_cl[mat_cl$column_id_ori == mat_columnanno[i,'column'], 'column_id'][1]
        } else {
            d <- abs(mat_cl$column_id_ori - mat_columnanno[i,'column'])
            if (min(d) <= min_d) {
                column_mid = mat_cl[mat_cl$column_id_ori == mat_cl$column_id_ori[which.min(d)], 'column_id'][1]
            }
            else {
                print(paste0('Error! min(d) is ', as.character(min(d))))
                #return() # influence 115,127,71,74
                next()
            }
        }
        start <- as.numeric(column_mid)
        end <- as.numeric(column_mid)
        sob@meta.data$gs[(sob@meta.data$column_id <= end) & (sob@meta.data$column_id >= start)] <- as.character(mat_columnanno[i,'g.s'])
        sob@meta.data$area[(sob@meta.data$column_id <= end) & (sob@meta.data$column_id >= start)] <- as.character(mat_columnanno[i,'area'])
        sob@meta.data$areaid[(sob@meta.data$column_id <= end) & (sob@meta.data$column_id >= start)] <- i
    }
    
    sob <- subset(sob, areaid != 'NA')
    res <- do.call(rbind, lapply(unique(sob$areaid), function(item) {
        s <- subset(sob, areaid == item)
        rbind(data.frame(sampletag = sampletag, gs = s$gs[1], areaid = item, cellnum = ncol(s), area = s$area[1], group ='all'),
            do.call(rbind, lapply(read.delim('/home/baiyq/projects/brain_folding/bert_data2024.v2/topct_model/top_features.inter30.csv',sep = ',')$celltype, function(ct) {
                if (ct %in% s$celltype) {
                    s <- subset(s, celltype == ct)
                    data.frame(sampletag = sampletag, gs = s$gs[1], areaid = item, cellnum = ncol(s), area = s$area[1], group =ct)
                } else {
                    data.frame(sampletag = sampletag, gs = s$gs[1], areaid = item, cellnum = NA, area = s$area[1], group = ct)
                }
            }))
        )
    }))
    write.csv(res, paste0('data2024.v2/gs_celldensity_1202/gs_cell_density.', sampletag, '.csv'), row.names = FALSE, quote = FALSE)
    return()
}

samplelist1 <- read.delim('unique.samplelist.txt')[,1]
samplelist2 <- read.delim('samplelist_test.txt')[,1]
res <- lapply(c(samplelist1, samplelist2), function(sampletag) {
    get_celldense(sampletag, min_d = 5)
})
