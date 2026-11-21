set.seed(123456789)
n <- 10
S <- 100
y <- rnorm(n)
x1 <- matrix(rnorm(n * S), nrow = S)
ll <- matrix(rnorm(n * S) * 0.1 - 1, nrow = S)
capacity <- rnorm(n)

# Discrete draws guarantee tied values and many zero excesses after thresholding
x_ties <- matrix(sample(seq(-2, 2, by = 0.5), n * S, replace = TRUE), nrow = S)
y_ties <- sample(seq(-2, 2, by = 0.5), n, replace = TRUE)

# Brute-force reference for the weighted Gini mean difference: full S x S
# double sum over independent draws (including s == t).
brute_weighted_gini <- function(v, w) {
  w <- w / sum(w)
  sum(w * t(w * t(outer(v, v, function(a, b) abs(a - b)))))
}

test_that("loo_excess_crps returns the documented structure", {
  out <- suppressWarnings(loo_excess_crps(x1, y, capacity, ll, cores = 1))
  expect_s3_class(out, "loo_excess_crps")
  expect_named(out, c("estimates", "pointwise", "diagnostics",
                      "psis_object", "capacity"))
  expect_named(out$estimates, c("Estimate", "SE"))
  expect_equal(colnames(out$pointwise), c("excess_crps", "EXX", "EXy"))
  expect_equal(nrow(out$pointwise), n)
  expect_length(out$diagnostics$pareto_k, n)
  expect_length(out$diagnostics$n_eff, n)
  expect_length(out$capacity, n)
  expect_s3_class(out$psis_object, "psis")
  # diagnostics accessors work on the returned object
  expect_equal(pareto_k_values(out), out$diagnostics$pareto_k)
  expect_equal(psis_n_eff_values(out), out$diagnostics$n_eff)
})

test_that("pointwise score has the CRPS orientation (larger is better)", {
  out <- suppressWarnings(loo_excess_crps(x1, y, capacity, ll, cores = 1))
  pw <- out$pointwise
  expect_equal(pw[, "excess_crps"], 0.5 * pw[, "EXX"] - pw[, "EXy"],
               ignore_attr = TRUE)
  expect_equal(out$estimates$Estimate, mean(pw[, "excess_crps"]))
  expect_equal(out$estimates$SE, sd(pw[, "excess_crps"]) / sqrt(n))
})

test_that("EXX matches the O(S^2) weighted double sum with ties and zeros", {
  out <- suppressWarnings(loo_excess_crps(x_ties, y_ties, capacity, ll,
                                          cores = 1))
  w <- weights(out$psis_object, log = FALSE, normalize = TRUE)
  for (i in seq_len(n)) {
    ex <- pmax(x_ties[, i] - capacity[i], 0)
    expect_equal(out$pointwise[i, "EXX"],
                 brute_weighted_gini(ex, w[, i]),
                 tolerance = 1e-12, ignore_attr = TRUE)
    ey <- max(y_ties[i] - capacity[i], 0)
    expect_equal(out$pointwise[i, "EXy"], sum(w[, i] * abs(ex - ey)),
                 tolerance = 1e-12, ignore_attr = TRUE)
  }
})

test_that("weighted_gini_mean_diff is exact for ties, weights and zeros", {
  v <- c(0, 0, 0, 1, 1, 2)
  w <- c(0.1, 0.2, 0.1, 0.25, 0.15, 0.2)
  expect_equal(weighted_gini_mean_diff(v, w),
               brute_weighted_gini(v, w),
               tolerance = 1e-14)
  # a block of zeros with concentrated weight
  v2 <- c(rep(0, 20), rep(2, 5))
  w2 <- c(rep(1, 20), c(10, 10, 1, 1, 1))
  w2 <- w2 / sum(w2)
  expect_equal(weighted_gini_mean_diff(v2, w2),
               brute_weighted_gini(v2, w2),
               tolerance = 1e-14)
  # constant and all-zero inputs are exactly zero
  w3 <- runif(8); w3 <- w3 / sum(w3)
  expect_equal(weighted_gini_mean_diff(rep(3.7, 8), w3), 0,
               tolerance = 1e-14, ignore_attr = TRUE)
  w4 <- runif(10); w4 <- w4 / sum(w4)
  expect_equal(weighted_gini_mean_diff(numeric(10), w4), 0,
               tolerance = 1e-14, ignore_attr = TRUE)
  # single draw is zero
  expect_equal(weighted_gini_mean_diff(1.3, 1), 0)
})

test_that("uniform weights recover the ordinary empirical CRPS", {
  ll_flat <- matrix(0, S, n)
  out <- suppressWarnings(loo_excess_crps(x1, y, capacity, ll_flat,
                                          cores = 1))
  w <- rep(1 / S, S)
  ref <- numeric(n)
  for (i in seq_len(n)) {
    ex <- pmax(x1[, i] - capacity[i], 0)
    ey <- max(y[i] - capacity[i], 0)
    EXX <- sum(outer(ex, ex, function(a, b) abs(a - b))) / S^2
    EXy <- mean(abs(ex - ey))
    ref[i] <- 0.5 * EXX - EXy
  }
  expect_equal(out$pointwise[, "excess_crps"], ref, tolerance = 1e-12)
})

test_that("capacity above all values yields exactly zero scores", {
  high <- max(x1, y) + 10
  out <- suppressWarnings(loo_excess_crps(x1, y, high, ll, cores = 1))
  expect_equal(max(abs(out$pointwise)), 0, ignore_attr = TRUE)
  expect_equal(out$estimates$Estimate, 0)
})

test_that("a single scalar capacity is applied to every observation", {
  out1 <- suppressWarnings(loo_excess_crps(x1, y, 0, ll, cores = 1))
  out2 <- suppressWarnings(loo_excess_crps(x1, y, rep(0, n), ll, cores = 1))
  expect_equal(out1$pointwise, out2$pointwise)
  expect_equal(out1$capacity, rep(0, n))
})

test_that("numeric method for one observation matches the matrix method", {
  m <- suppressWarnings(
    loo_excess_crps(x1[, 1, drop = FALSE], y[1], capacity[1],
                    ll[, 1, drop = FALSE], cores = 1)
  )
  v <- suppressWarnings(
    loo_excess_crps(x1[, 1], y[1], capacity[1], ll[, 1], cores = 1)
  )
  expect_equal(v$pointwise, m$pointwise)
})

test_that("input validation throws correct errors", {
  expect_error(
    loo_excess_crps(x1, y, capacity, matrix(0, S, n + 1), cores = 1),
    "identical(dim(log_lik), dim(x)) is not TRUE",
    fixed = TRUE
  )
  expect_error(
    loo_excess_crps(x1, y, capacity[1:2], ll, cores = 1),
    "length(capacity) == 1 || length(capacity) == length(y) is not TRUE",
    fixed = TRUE
  )
  expect_error(
    loo_excess_crps(x1, y[1:5], capacity, ll, cores = 1),
    "ncol(x) == length(y) is not TRUE",
    fixed = TRUE
  )
  expect_error(
    loo_excess_crps(as.character(x1), y, capacity, ll, cores = 1),
    "no applicable method"
  )
  expect_error(
    validate_excess_crps_input(x1, y, as.character(capacity), ll),
    "is.numeric(capacity) is not TRUE",
    fixed = TRUE
  )
})

test_that("legacy loo_crps entry point is unchanged", {
  set.seed(1)
  x2 <- x1[sample(S), ]
  out <- suppressWarnings(loo_crps(x1, x2, y, ll, cores = 1))
  expect_named(out, c("estimates", "pointwise"))
  expect_length(out$pointwise, n)
  expect_equal(names(out$estimates), c("Estimate", "SE"))
  # non-LOO crps still needs two draw matrices
  expect_named(crps(x1, x2, y), c("estimates", "pointwise"))
})
