#' Continuously ranked probability score
#'
#' The `crps()` and `scrps()` functions and their `loo_*()` counterparts can be
#' used to compute the continuously ranked probability score (CRPS) and scaled
#' CRPS (SCRPS) (as defined by Bolin and Wallin, 2023). CRPS is a proper scoring rule, and
#' strictly proper when the first moment of the predictive distribution is
#' finite. Both can be expressed in terms of samples form the predictive
#' distribution. See, for example, a paper by Gneiting and Raftery (2007)
#' for a comprehensive discussion on CRPS.
#'
#' To compute (S)CRPS, the user needs to provide two sets of draws, `x` and
#' `x2`, from the predictive distribution. This is due to the fact that formulas
#' used to compute CRPS involve an expectation of the absolute difference of `x`
#' and `x2`, both having the same distribution. See the `permutations` argument,
#' as well as Gneiting and Raftery (2007) for details.
#'
#' @export
#' @param x A `S` by `N` matrix (draws by observations), or a vector of length
#'   `S` when only single observation is provided in `y`.
#' @param x2 Independent draws from the same distribution as draws in `x`.
#'   Should be of the identical dimension.
#' @param y A vector of observations or a single value.
#' @param permutations An integer, with default value of 1,  specifying how many
#'   times the expected value of  |X - X'| (`|x - x2|`) is computed. The row
#'   order of `x2` is shuffled as elements `x` and `x2` are typically drawn
#'   given the same values of parameters. This happens, e.g., when one calls
#'   `posterior_predict()` twice for a fitted \pkg{rstanarm} or \pkg{brms}
#'   model. Generating more permutations is expected to decrease the variance of
#'   the computed expected value.
#' @param ... Passed on to [E_loo()] in the `loo_*()` version of these
#'   functions.
#'
#' @return A list containing two elements: `estimates` and `pointwise`.
#'   The former reports estimator and standard error and latter the pointwise
#'   values. Following Bolin & Wallin (2023), a larger value is better.
#'
#' @examples
#' \dontrun{
#' # An example using rstanarm
#' library(rstanarm)
#' data("kidiq")
#' fit <- stan_glm(kid_score ~ mom_hs + mom_iq, data = kidiq)
#' ypred1 <- posterior_predict(fit)
#' ypred2 <- posterior_predict(fit)
#' crps(ypred1, ypred2, y = fit$y)
#' loo_crps(ypred1, ypred2, y = fit$y, log_lik = log_lik(fit))
#' }
#'
#' @references
#' Bolin, D., & Wallin, J. (2023). Local scale invariance and robustness of
#' proper scoring rules. Statistical Science, 38(1):140-159.
#'
#' Gneiting, T., & Raftery, A. E. (2007). Strictly Proper Scoring Rules,
#' Prediction, and Estimation. Journal of the American Statistical Association,
#' 102(477), 359–378.
crps <- function(x, ...) {
  UseMethod("crps")
}


#' @rdname crps
#' @export
scrps <- function(x, ...) {
  UseMethod("scrps")
}

#' @rdname crps
#' @export
loo_crps <- function(x, ...) {
  UseMethod("loo_crps")
}

#' @rdname crps
#' @export
loo_scrps <- function(x, ...) {
  UseMethod("loo_scrps")
}


#' Leave-one-out CRPS for excess over a capacity threshold
#'
#' @description
#' The `loo_excess_crps()` function computes a leave-one-out (LOO)
#' continuously ranked probability score (CRPS) for the amount by which a
#' quantity exceeds an observation-specific capacity (for example, how much an
#' inventory exceeds the capacity available on a given day). Both the
#' observations and the predictive draws are first transformed into excesses
#' over the capacity using
#' \deqn{e(z) = \max(z - c, 0),}
#' where `c` is the capacity, so that values at or below the capacity are
#' recorded as zero. CRPS is then evaluated on these (typically
#' zero-inflated) excesses.
#'
#' Unlike [loo_crps()], `loo_excess_crps()` requires only a single set of
#' predictive draws `x` together with the matching log-likelihood matrix
#' `log_lik`. The expected absolute difference between two independent draws
#' from the LOO predictive distribution is computed analytically from the
#' weighted empirical distribution of `x` using only sorting and cumulative
#' sums, which is exact in the presence of tied values and of the (possibly
#' many) zero excesses. No random pairing of draws and no pairwise matrices are
#' used.
#'
#' For observation `i`, with normalized PSIS weights \eqn{w_{i,s}} and excess
#' draws \eqn{e_{i,s} = \max(x_{s,i} - c_i, 0)}, the two terms of CRPS are
#' \deqn{E_w|e(X) - e(y_i)| = \sum_s w_{i,s} |e_{i,s} - e(y_i)|}
#' and
#' \deqn{E_w|e(X) - e(X')| = \sum_s \sum_t w_{i,s} w_{i,t}
#'       |e_{i,s} - e_{i,t}|,}
#' with `X` and `X'` two independent draws from the same weighted
#' distribution. Following the orientation used by [crps()] and [loo_crps()]
#' (Bolin and Wallin, 2023), the returned pointwise score is
#' \deqn{\frac{1}{2} E_w|e(X) - e(X')| - E_w|e(X) - e(y_i)|,}
#' which is a proper score for which a larger value is better.
#'
#' @export
#' @param x An `S` by `N` matrix (draws by observations) of predictive
#'   samples, or a vector of length `S` when a single observation is provided
#'   in `y`.
#' @param y A vector of observed values (one per observation), or a single
#'   value.
#' @param capacity A numeric vector of observation-specific capacities of the
#'   same length as `y`, or a single value used for every observation.
#' @param log_lik A log-likelihood matrix the same size as `x`.
#' @param r_eff An optional vector of relative effective sample size estimates
#'   containing one element per observation. See [psis()] for details.
#' @param cores The number of cores to use for parallelization of [psis()].
#'   See [psis()] for details.
#' @param ... Passed on to [psis()].
#'
#' @return A list of class `"loo_excess_crps"` containing:
#' \describe{
#'   \item{`estimates`}{
#'   A data frame with the aggregate `Estimate` (mean of the pointwise scores)
#'   and its standard error `SE` (`sd(pointwise) / sqrt(N)`).
#'   }
#'   \item{`pointwise`}{
#'   A matrix with one row per observation and the columns
#'   `excess_crps` (the pointwise score), `EXX` (the weighted expected
#'   absolute difference of two independent excess draws), and `EXy` (the
#'   weighted expected absolute difference to the observed excess).
#'   }
#'   \item{`diagnostics`}{
#'   A list with the per-observation PSIS `pareto_k` estimates and PSIS
#'   effective sample sizes `n_eff`, copied from the underlying [psis()]
#'   object. See [pareto-k-diagnostic] for how to interpret them.
#'   }
#'   \item{`psis_object`}{
#'   The full [psis()] object used to obtain the weights.
#'   }
#'   \item{`capacity`}{
#'   The capacities used, recycled to length `N`.
#'   }
#' }
#' The PSIS diagnostics can also be extracted with [pareto_k_values()] and
#' [psis_n_eff_values()].
#'
#' @examples
#' \dontrun{
#' # An example using rstanarm
#' library(rstanarm)
#' data("kidiq")
#' fit <- stan_glm(kid_score ~ mom_hs + mom_iq, data = kidiq)
#' ypred <- posterior_predict(fit)
#' capacities <- quantile(fit$y, 0.8)
#' loo_excess_crps(ypred, fit$y, capacities, log_lik = log_lik(fit))
#' }
#'
#' @references
#' Bolin, D., & Wallin, J. (2023). Local scale invariance and robustness of
#' proper scoring rules. Statistical Science, 38(1):140-159.
#'
#' Gneiting, T., & Raftery, A. E. (2007). Strictly Proper Scoring Rules,
#' Prediction, and Estimation. Journal of the American Statistical Association,
#' 102(477), 359–378.
loo_excess_crps <- function(x, ...) {
  UseMethod("loo_excess_crps")
}

#' @rdname loo_excess_crps
#' @export
loo_excess_crps.matrix <-
  function(x,
           y,
           capacity,
           log_lik,
           ...,
           r_eff = 1,
           cores = getOption("mc.cores", 1)) {
    validate_excess_crps_input(x, y, capacity, log_lik)
    N <- ncol(x)
    capacity <- rep_len(capacity, N)

    # Excess over each observation's capacity: max(value - capacity, 0).
    excess_x <- pmax(sweep(x, 2, capacity, check.margin = FALSE), 0)
    excess_y <- pmax(y - capacity, 0)

    psis_object <- psis(-log_lik, r_eff = r_eff, cores = cores, ...)
    w <- weights(psis_object, log = FALSE, normalize = TRUE)

    EXy <- colSums(w * abs(sweep(excess_x, 2, excess_y, check.margin = FALSE)))
    EXX <- vapply(
      seq_len(N),
      function(i) weighted_gini_mean_diff(excess_x[, i], w[, i]),
      FUN.VALUE = numeric(1)
    )

    pointwise <- cbind(
      excess_crps = .crps_fun(EXX, EXy),
      EXX = EXX,
      EXy = EXy
    )
    excess_crps_output(pointwise, psis_object, capacity)
  }

#' Method for a single data point
#' @rdname loo_excess_crps
#' @export
loo_excess_crps.numeric <-
  function(x,
           y,
           capacity,
           log_lik,
           ...,
           r_eff = 1,
           cores = getOption("mc.cores", 1)) {
    stopifnot(length(y) == 1)
    loo_excess_crps.matrix(
      as.matrix(x),
      y = y,
      capacity = capacity,
      log_lik = as.matrix(log_lik),
      ...,
      r_eff = r_eff,
      cores = cores
    )
  }


# ------------ Internals ----------------


#' Weighted expected absolute difference of two independent draws
#'
#' Computes \eqn{\sum_{s,t} w_s w_t |v_s - v_t|} for normalized weights
#' \eqn{w_s} using only sorting and cumulative sums, so the result is exact
#' even with tied values and (many) zeros, and avoids an `S x S` matrix.
#'
#' @noRd
#' @param v Numeric vector of draw values (e.g. excesses over a capacity).
#' @param w Numeric vector of non-negative weights, normalized to sum to one.
#' @return A scalar.
weighted_gini_mean_diff <- function(v, w) {
  ord <- order(v)
  v <- v[ord]
  w <- w[ord]
  # Cumulative weight strictly before each position. Positions sharing a value
  # get identical coefficients, so ties (including a block of zeros) contribute
  # nothing for within-block pairs, exactly as in the double sum.
  w_cum <- cumsum(w)
  w_before <- c(0, w_cum[-length(w_cum)])
  coeff <- 2 * w_before + w - 1 # weight below this value minus weight above it
  # The factor 2 turns the sum over unordered pairs into the ordered double sum.
  2 * sum(v * w * coeff)
}

#' Compute output object for `loo_excess_crps()`
#' @noRd
excess_crps_output <- function(pointwise, psis_object, capacity) {
  n <- nrow(pointwise)
  score <- pointwise[, "excess_crps"]
  estimates <- data.frame(
    Estimate = mean(score),
    SE = stats::sd(score) / sqrt(n)
  )
  structure(
    list(
      estimates = estimates,
      pointwise = pointwise,
      diagnostics = list(
        pareto_k = pareto_k_values(psis_object),
        n_eff = psis_n_eff_values(psis_object)
      ),
      psis_object = psis_object,
      capacity = capacity
    ),
    class = "loo_excess_crps"
  )
}

#' Validate input of `loo_excess_crps()`
#' @noRd
validate_excess_crps_input <- function(x, y, capacity, log_lik) {
  stopifnot(
    is.numeric(x),
    is.numeric(y),
    is.numeric(capacity),
    is.numeric(log_lik),
    ncol(x) == length(y),
    length(capacity) == 1 || length(capacity) == length(y),
    identical(dim(log_lik), dim(x))
  )
  invisible(TRUE)
}

#' Print method for `loo_excess_crps()` output
#'
#' @export
#' @param x An object returned by [loo_excess_crps()].
#' @param digits An integer passed to [base::round()].
#' @param ... Ignored.
#' @return `x`, invisibly.
print.loo_excess_crps <- function(x, digits = 1, ...) {
  cat("\n")
  cat(
    "Computed from", dim(x$psis_object)[1], "posterior draws and",
    dim(x$psis_object)[2], "observations using PSIS.\n"
  )
  cat("LOO excess CRPS (larger is better):\n\n")
  print(.fr(as.data.frame(x$estimates), digits), quote = FALSE)
  cat("\n")
  print(pareto_k_table(x$psis_object), digits = digits)
  cat(.k_help())
  invisible(x)
}


#' @rdname crps
#' @export
crps.matrix <- function(x, x2, y, ..., permutations = 1) {
  validate_crps_input(x, x2, y)
  repeats <- replicate(permutations, EXX_compute(x, x2), simplify = F)
  EXX <- Reduce(`+`, repeats) / permutations
  EXy <- colMeans(abs(sweep(x, 2, y)))
  crps_output(.crps_fun(EXX, EXy))
}


#' Method for a single data point
#' @rdname crps
#' @export
crps.numeric <- function(x, x2, y, ..., permutations = 1) {
  stopifnot(length(x) == length(x2),
            length(y) == 1)
  crps.matrix(as.matrix(x), as.matrix(x2), y, permutations = permutations)
}


#' @rdname crps
#' @export
#' @param log_lik A log-likelihood matrix the same size as `x`.
#' @param r_eff An optional vector of relative effective sample size estimates
#'   containing one element per observation. See [psis()] for details.
#' @param cores The number of cores to use for parallelization of `[psis()]`.
#'   See [psis()] for details.
loo_crps.matrix <-
  function(x,
           x2,
           y,
           log_lik,
           ...,
           permutations = 1,
           r_eff = 1,
           cores = getOption("mc.cores", 1)) {
  validate_crps_input(x, x2, y, log_lik)
  repeats <- replicate(permutations,
                       EXX_loo_compute(x, x2, log_lik, r_eff = r_eff, ...),
                       simplify = F)
  EXX <- Reduce(`+`, repeats) / permutations
  psis_obj <- psis(-log_lik, r_eff = r_eff, cores = cores)
  EXy <- E_loo(abs(sweep(x, 2, y)), psis_obj, log_ratios = -log_lik, ...)$value
  crps_output(.crps_fun(EXX, EXy))
}


#' @rdname crps
#' @export
scrps.matrix <- function(x, x2, y, ..., permutations = 1) {
  validate_crps_input(x, x2, y)
  repeats <- replicate(permutations, EXX_compute(x, x2), simplify = F)
  EXX <- Reduce(`+`, repeats) / permutations
  EXy <- colMeans(abs(sweep(x, 2, y)))
  crps_output(.crps_fun(EXX, EXy, scale = TRUE))
}

#' @rdname crps
#' @export
scrps.numeric <- function(x, x2, y, ..., permutations = 1) {
  stopifnot(length(x) == length(x2),
            length(y) == 1)
  scrps.matrix(as.matrix(x), as.matrix(x2), y, permutations = permutations)
}


#' @rdname crps
#' @export
loo_scrps.matrix <-
  function(
    x,
    x2,
    y,
    log_lik,
    ...,
    permutations = 1,
    r_eff = 1,
    cores = getOption("mc.cores", 1)) {
  validate_crps_input(x, x2, y, log_lik)
  repeats <- replicate(permutations,
                       EXX_loo_compute(x, x2, log_lik, r_eff = r_eff, ...),
                       simplify = F)
  EXX <- Reduce(`+`, repeats) / permutations
  psis_obj <- psis(-log_lik, r_eff = r_eff, cores = cores)
  EXy <- E_loo(abs(sweep(x, 2, y)), psis_obj, log_ratios = -log_lik, ...)$value
  crps_output(.crps_fun(EXX, EXy, scale = TRUE))
}

# ------------ Internals ----------------


EXX_compute <- function(x, x2) {
  S <- nrow(x)
  colMeans(abs(x - x2[sample(1:S),]))
}


EXX_loo_compute <- function(x, x2, log_lik, r_eff = 1, ...) {
  S <- nrow(x)
  shuffle <- sample (1:S)
  x2 <- x2[shuffle,]
  log_lik2 <- log_lik[shuffle,]
  psis_obj_joint <- psis(-log_lik - log_lik2 , r_eff = r_eff)
  E_loo(abs(x - x2), psis_obj_joint, log_ratios = -log_lik - log_lik2, ...)$value
}


#' Function to compute crps and scrps
#' @noRd
.crps_fun <- function(EXX, EXy, scale = FALSE) {
  if (scale) return(-EXy/EXX - 0.5 * log(EXX))
  0.5 * EXX - EXy
}

#' Compute output data for crps functions
#' @noRd
crps_output <- function(crps_pw) {
  n <- length(crps_pw)
  out <- list()
  out$estimates <- c(mean(crps_pw), sd(crps_pw) / sqrt(n))
  names(out$estimates) <- c('Estimate', 'SE')
  out$pointwise <- crps_pw
  out
}

#' Validate input of CRPS functions
#'
#' Check that predictive draws and observed data are of compatible shape
#' @noRd
validate_crps_input <- function(x, x2, y, log_lik = NULL) {
  stopifnot(is.numeric(x),
            is.numeric(x2),
            is.numeric(y),
            identical(dim(x), dim(x2)),
            ncol(x) == length(y),
            ifelse(is.null(log_lik), TRUE, identical(dim(log_lik), dim(x)))
            )
}
