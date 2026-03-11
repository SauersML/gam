library(mgcv)
set.seed(42)
n <- 200
df <- data.frame(
    y = rnorm(n),
    x1 = runif(n),
    x2 = runif(n),
    x3 = runif(n)
)

fits <- tryCatch(gam(list(y ~ s(x1, bs="ps") + s(x2, bs="ps"), ~ s(x2, bs="ps") + s(x3, bs="ps")), family=gaulss(), data=df, method="REML", select=TRUE), error=function(e) e)
print(fits)

fitns <- tryCatch(gam(list(y ~ s(x1, bs="ps") + s(x2, bs="ps"), ~ s(x2, bs="ps") + s(x3, bs="ps")), family=gaulss(), data=df, method="REML", select=FALSE), error=function(e) e)
print(class(fitns))
