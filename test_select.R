library(mgcv)
set.seed(42)
n <- 200
df <- data.frame(
    y = rnorm(n),
    x1 = runif(n),
    x2 = runif(n)
)

fits <- gam(list(y ~ s(x1, bs="ps", k=20) + s(x2, bs="ps", k=20), ~ s(x1, bs="ps", k=20) + s(x2, bs="ps", k=20)), family=gaulss(), data=df, select=TRUE)
print(class(fits))
