import RealModule

public extension Collection {
  /// Performs a one-sample t-test on the selected variable.
  /// - parameter variable: The variable under test.
  /// - parameter mu0: The hypothesized population mean.
  /// - returns: A `HypothesisTestResult` containing the t-statistic, p-value, and degrees of freedom.
  ///
  /// Tests whether the sample mean of the variable differs significantly from a hypothesized value.
  /// This is a two-tailed test. The collection must contain at least two elements;
  /// otherwise the result contains NaN values.
  /// The time complexity of this method is O(n).
  func oneSampleTTest<T: ConvertibleToReal>(
    variable: KeyPath<Element, T>,
    hypothesizedMean mu0: Double
  ) -> HypothesisTestResult {

    let n = self.count.realValue
    guard n >= 2 else {
      return HypothesisTestResult(
        testStatistic: .signalingNaN,
        pValue: .signalingNaN,
        degreesOfFreedom: .signalingNaN
      )
    }

    let sampleMean = self.mean(variable: variable)
    let sampleSD = self.standardDeviation(variable: variable, from: .sample)
    let standardError = sampleSD / n.squareRoot()

    guard standardError > 0 else {
      // All values identical: if they equal mu0 then t = 0, p = 1; otherwise infinite t, p = 0.
      let t: Double = sampleMean == mu0 ? 0 : .infinity
      let p: Double = sampleMean == mu0 ? 1 : 0
      return HypothesisTestResult(
        testStatistic: t,
        pValue: p,
        degreesOfFreedom: n - 1
      )
    }

    let t = (sampleMean - mu0) / standardError
    let df = n - 1
    let pValue = twoTailedPValue(t: t, df: df)

    return HypothesisTestResult(
      testStatistic: t,
      pValue: pValue,
      degreesOfFreedom: df
    )
  }

  /// Performs Welch's two-sample independent t-test.
  /// - parameter sample1: The first sample collection.
  /// - parameter sample2: The second sample collection.
  /// - parameter variable: The variable under test.
  /// - returns: A `HypothesisTestResult` containing the t-statistic, p-value, and degrees of freedom.
  ///
  /// Tests whether the means of two independent samples differ significantly.
  /// Uses the Welch-Satterthwaite approximation for degrees of freedom (unequal variances).
  /// This is a two-tailed test. Each sample must contain at least two elements;
  /// otherwise the result contains NaN values.
  static func twoSampleTTest<C: Collection, T: ConvertibleToReal>(
    _ sample1: C,
    _ sample2: C,
    variable: KeyPath<C.Element, T>
  ) -> HypothesisTestResult
  where C.Element == Element
  {
    let n1 = sample1.count.realValue
    let n2 = sample2.count.realValue

    guard n1 >= 2, n2 >= 2 else {
      return HypothesisTestResult(
        testStatistic: .signalingNaN,
        pValue: .signalingNaN,
        degreesOfFreedom: .signalingNaN
      )
    }

    let mean1 = sample1.mean(variable: variable)
    let mean2 = sample2.mean(variable: variable)
    let var1 = sample1.variance(variable: variable, from: .sample)
    let var2 = sample2.variance(variable: variable, from: .sample)

    let se1 = var1 / n1
    let se2 = var2 / n2
    let pooledSE = (se1 + se2).squareRoot()

    guard pooledSE > 0 else {
      let t: Double = mean1 == mean2 ? 0 : .infinity
      let p: Double = mean1 == mean2 ? 1 : 0
      return HypothesisTestResult(
        testStatistic: t,
        pValue: p,
        degreesOfFreedom: n1 + n2 - 2
      )
    }

    let t = (mean1 - mean2) / pooledSE

    // Welch-Satterthwaite degrees of freedom
    let numerator = (se1 + se2) * (se1 + se2)
    let denominator = (se1 * se1) / (n1 - 1) + (se2 * se2) / (n2 - 1)
    let df = numerator / denominator

    let pValue = twoTailedPValue(t: t, df: df)

    return HypothesisTestResult(
      testStatistic: t,
      pValue: pValue,
      degreesOfFreedom: df
    )
  }
}

/// Computes the two-tailed p-value for a t-test using the regularized incomplete beta function.
/// - parameter t: The t-statistic.
/// - parameter df: The degrees of freedom.
/// - returns: The two-tailed p-value.
///
/// Uses the identity: p = I(df / (df + t²); df/2, 1/2)
/// where I is the regularized incomplete beta function.
@usableFromInline
internal func twoTailedPValue(t: Double, df: Double) -> Double {
  let tSquared = t * t

  // When t is zero the p-value is exactly 1 (no evidence against H0).
  guard tSquared > 0 else { return 1 }

  let x = df / (df + tSquared)

  // When x is at the boundary (extremely large |t|), p-value is effectively 0.
  guard x > 0, x < 1 else { return 0 }

  return regularizedIncompleteBeta(x: x, alpha: df / 2, beta: 0.5)
}
