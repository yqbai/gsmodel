suppressMessages(library(data.table))
suppressMessages(library(readxl))
suppressMessages(library(Seurat))
suppressMessages(library(reshape2))
suppressMessages(library(dplyr))

setwd('data2024.v2/')

rename2 <- fread(paste0('anno.264clusters.renameV1212.txt'), header = TRUE, sep = "\t", data.table = F)
rownames(rename2) <- rename2$anno2

mat_marker <- readRDS('markers.264clusters.FinalName.rds')
mat_ftimp <- read.delim('feature_importances/feature_importances.bert.txt')
mat_ftimp$FullName <- rename2$FullName

load_sob <- function(sampletag, n_extend_plan = 50, min_d = 5) {
    mat_cl <- fread(paste0('spa_matrix/cell-meta-matrix_1105/cell-meta-matrix.', sampletag, '.tsv'), header = TRUE, sep = "\t", data.table = F)
    dim(mat_cl)
    mat_cl$layer <- unlist(lapply(mat_cl$gene_area, function(item) strsplit(item, split = '[-]')[[1]][3]))
    mat_cl$region <- unlist(lapply(mat_cl$gene_area, function(item) strsplit(item, split = '[-]')[[1]][2]))
    mat_cl$ry <- 0 - mat_cl$ry

    mat_cl$class <- lapply(rename2[mat_cl$cluster, 'FullName'], function(item) strsplit(item, split = '[ ]')[[1]][1])
    mat_cl$subclass <- lapply(rename2[mat_cl$cluster, 'FullName'], function(item) strsplit(strsplit(item, split = '[ ]')[[1]][2], split = '[.]')[[1]][1])
    mat_cl$celltype <- rename2[mat_cl$cluster, 'FullName']

    mat_column <- fread(paste0('data2024/cell_column/', sampletag, '_cell2column.csv'), header = TRUE, sep = ",", data.table = F)
    mat_cl <- merge(mat_cl, mat_column, by = 'cell_id')
    print(dim(mat_cl))

    colnames(mat_cl)[colnames(mat_cl) == 'column_id'] <- 'column_id_ori'
    a <- data.frame(row.names = sort(unique(mat_cl$column_id_ori)), reid = 1:length(unique(mat_cl$column_id_ori)))
    mat_cl$column_id <- a[as.character(mat_cl$column_id_ori), 'reid']

    sob <- readRDS(paste0('spa_matrix/spa_matrix.', sampletag, '.rds'))
    sob <- subset(sob, cells = as.character(mat_cl$cell_id))
    sob@meta.data <- mat_cl
    rownames(sob@meta.data) <- mat_cl$cell_id
    table(rownames(sob@meta.data) == colnames(sob))
    table(rownames(sob@meta.data) == mat_cl$cell_id)
    
    mat_columnanno <- read.delim('data2024/column2annot_man_res_corrected.csv', sep = ',', header = T)
    mat_columnanno <- mat_columnanno[mat_columnanno$section == sampletag,]
    mat_columnanno <- mat_columnanno[order(mat_columnanno$column),]

    if (nrow(mat_columnanno) == 1) {
        n_extend <- n_extend_plan
    } else {
        n_extend <- floor(min(unlist(lapply(2:nrow(mat_columnanno), function(i) mat_columnanno$column[i] - mat_columnanno$column[i-1]))/2, n_extend_plan))
    }
    print(n_extend)
    
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
                return()
            }
        }
        start <- as.numeric(column_mid) - n_extend
        end <- as.numeric(column_mid) + n_extend
        sob@meta.data$gs[(sob@meta.data$column_id <= end) & (sob@meta.data$column_id >= start)] <- as.character(mat_columnanno[i,'g.s'])
        sob@meta.data$area[(sob@meta.data$column_id <= end) & (sob@meta.data$column_id >= start)] <- as.character(mat_columnanno[i,'area'])
    }
    return(sob)
}

getmode <- function(v) {
   uniqv <- unique(v[(!is.na(v)) & (v!='unknown')])
   uniqv[which.max(tabulate(match(v, uniqv)))]
}

get_bulk_sob <- function(sob, agg_tag) {
    s <- subset(sob, gs != 'NA')
    res <- lapply(unique(s@meta.data[,agg_tag]), function(id) {
        s <- subset(s, cells = colnames(s)[s@meta.data[,agg_tag] == id])
        df <- apply(s@assays$SCT@data, 1, mean)
        return(list(df, s$gs[1], getmode(s$region), id))
    })
    df <- do.call(cbind, lapply(res, function(item) item[[1]]))
    s_bulk <- CreateSeuratObject(df)
    s_bulk@meta.data$gs <- unlist(lapply(res, function(item) item[[2]]))
    s_bulk@meta.data$region <- unlist(lapply(res, function(item) item[[3]]))
    s_bulk@meta.data$sampletag <- sob@meta.data$sampletag[1]
    s_bulk@meta.data$agg_tag <- unlist(lapply(res, function(item) item[[4]]))
    return(s_bulk)
}

setwd('data2024.v2/bulk_column')

sob <- load_sob(sampletag)
s <- subset(sob, gs != 'NA')
s_bulk <- get_bulk_sob(s, agg_tag = 'column_id')
s_bulk@meta.data$gs_flag <- ifelse(s_bulk@meta.data$gs == '1', 'Gyrus', 'Sulcus')
s_bulk <- SetIdent(s_bulk, value = s_bulk@meta.data$gs_flag)
ref.markers <- FindAllMarkers(object = s_bulk, test.use = 'roc', slot = 'counts', assay = 'RNA', return.thresh = 0)
saveRDS(ref.markers, paste0('ref.markers.roc.', sampletag, '.rds'))

