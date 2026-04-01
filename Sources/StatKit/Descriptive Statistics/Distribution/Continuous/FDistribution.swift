import RealModule

/// A type modelling an F-Distribution (Fisher-Snedecor).
public struct FDistribution: ContinuousDistribution, UnivariateDistribution {
  /// The first degrees of freedom parameter (numerator).
  public let d1: Double

  /// The second degrees of freedom parameter (denominator).
  public let d2: Double

  /// A chi-squared distribution helper for the numerator degrees of freedom.
  private let chiSquared1: ChiSquaredDistribution

  /// A chi-squared distribution helper for the denominator degrees of freedom.
  private let chiSquared2: ChiSquaredDistribution

  /// Creates an F-Distribution with specified degrees of freedom.
  /// - parameter d1: The numerator degrees of freedom.
  /// - parameter d2: The denominator degrees of freedom.
  public init(d1: Double, d2: Double) {
    precondition(
      0 < d1,
      "The d1 parameter needs to be greater than 0 (\(d1) was used)."
    )
    precondition(
      0 < d2,
      "The d2 parameter needs to be greater than 0 (\(d2) was used)."
    )

    self.d1 = d1
    self.d2 = d2
    self.chiSquared1 = ChiSquaredDistribution(degreesOfFreedom: Int(d1))
    self.chiSquared2 = ChiSquaredDistribution(degreesOfFreedom: Int(d2))
  }

  public func pdf(x: Double, logarithmic: Bool = false) -> Double {
    guard 0 < x else {
      return logarithmic ? -.infinity : .zero
    }

    let logNumerator = (d1 / 2) * .log(d1) + (d2 / 2) * .log(d2)
      + (d1 / 2 - 1) * .log(x)
    let logDenominator = ((d1 + d2) / 2) * .log(d1 * x + d2)
      + StatKit.beta(alpha: d1 / 2, beta: d2 / 2, logarithmic: true)
    let logValue = logNumerator - logDenominator

    return logarithmic ? logValue : .exp(logValue)
  }

  public var mean: Double {
    return d2 > 2 ? d2 / (d2 - 2) : .nan
  }

  public var variance: Double {
    guard d2 > 4 else { return .nan }
    let numerator = 2 * d2 * d2 * (d1 + d2 - 2)
    let denominator = d1 * (d2 - 2) * (d2 - 2) * (d2 - 4)
    return numerator / denominator
  }

  public var skewness: Double {
    guard d2 > 6 else { return .nan }
    let numerator = (2 * d1 + d2 - 2) * .sqrt(8 * (d2 - 4))
    let denominator = (d2 - 6) * .sqrt(d1 * (d1 + d2 - 2))
    return numerator / denominator
  }

  public var excessKurtosis: Double {
    guard d2 > 8 else { return .nan }
    let numerator = 12 * (d1 * (5 * d2 - 22) * (d1 + d2 - 2) + (d2 - 4) * .pow(d2 - 2, 2))
    let denominator = d1 * (d2 - 6) * (d2 - 8) * (d1 + d2 - 2)
    return numerator / denominator
  }

  public func cdf(x: Double, logarithmic: Bool = false) -> Double {
    guard 0 < x else {
      return logarithmic ? -.infinity : .zero
    }

    let z = d1 * x / (d1 * x + d2)

    // Handle edge cases where z reaches boundary values.
    if z <= 0 {
      return logarithmic ? -.infinity : 0
    }
    if z >= 1 {
      return logarithmic ? 0 : 1
    }

    let result = regularizedIncompleteBeta(x: z, alpha: d1 / 2, beta: d2 / 2)

    switch result {
      case ...0:
        return logarithmic ? -.infinity : 0

      case 1...:
        return logarithmic ? 0 : 1

      default:
        return logarithmic ? .log(result) : result
    }
  }

  /// Samples a single value from the distribution.
  /// - returns: A sample from the distribution.
  ///
  /// Sampling uses the ratio F = (X1/d1) / (X2/d2),
  /// where X1 ~ Chi-Squared(d1) and X2 ~ Chi-Squared(d2).
  public func sample() -> Double {
    let x1 = chiSquared1.sample()
    let x2 = chiSquared2.sample()
    return (x1 / d1) / (x2 / d2)
  }

  /// Samples a specified number of values from the distribution.
  /// - parameter numberOfElements: The number of samples to generate.
  /// - returns: An array of sampled values.
  public func sample(_ numberOfElements: Int) -> [Double] {
    precondition(0 < numberOfElements, "The requested number of samples need to be greater than 0.")

    let x1Samples = chiSquared1.sample(numberOfElements)
    let x2Samples = chiSquared2.sample(numberOfElements)
    return zip(x1Samples, x2Samples).map { x1, x2 in
      (x1 / d1) / (x2 / d2)
    }
  }
}
