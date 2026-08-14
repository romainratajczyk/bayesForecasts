
# 1. Configuration du répertoire de librairie utilisateur
lib_path <- Sys.getenv("R_LIBS_USER")
dir.create(lib_path, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(lib_path, .libPaths()))

# 2. Installation conditionnelle locale
if (!requireNamespace("dbarts", quietly = TRUE)) {
    install.packages("dbarts", repos="https://cloud.r-project.org", lib=lib_path)
}
suppressMessages(library(dbarts))
set.seed(42)

Xb <- as.matrix(read.csv('bart_X_tr.csv', header=FALSE))
yb <- as.numeric(read.csv('bart_y_tr.csv', header=FALSE)[,1])
Xt <- as.matrix(read.csv('bart_X_te.csv', header=FALSE))

fit <- bart(x.train = Xb, y.train = yb, x.test = Xt,
            ntree = 200, k = 2.0, power = 2.0, base = 0.95,
            ndpost = 1000, nskip = 500, verbose = FALSE)

write.table(pnorm(fit$yhat.test), 'bart_p_te.csv', row.names=FALSE, col.names=FALSE, sep=',')
write.table(pnorm(fit$yhat.train), 'bart_p_tr.csv', row.names=FALSE, col.names=FALSE, sep=',')
