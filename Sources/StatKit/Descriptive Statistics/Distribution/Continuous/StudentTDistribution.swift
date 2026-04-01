import RealModule

/// A type modelling a Student's t-Distribution.
public struct StudentTDistribution: ContinuousDistribution, UnivariateDistribution {
  /// The distributions degrees of freedom parameter.
  public let degreesOfFreedom: Double

  /// A standard normal distribution helper for sampling.
  private let normal: NormalDistribution

  /// A chi-squared distribution helper for sampling.
  private let chiSquared: ChiSquaredDistribution

  /// Creates a Student's t-Distribution with a specified degrees of freedom.
  /// - parameter degreesOfFreedom: The distributions degrees of freedom parameter.
  public init(degreesOfFreedom: Double) {
    precondition(
      0 < degreesOfFreedom,
      "The degreesOfFreedom parameter needs to be greater than 0 (\(degreesOfFreedom) was used)."
    )

    self.degreesOfFreedom = degreesOfFreedom
    self.normal = NormalDistribution(mean: 0, variance: 1)
    self.chiSquared = ChiSquaredDistribution(degreesOfFreedom: Int(degreesOfFreedom))
  }

  public func pdf(x: Double, logarithmic: Bool = false) -> Double {
    let nu = degreesOfFreedom
    let logNumerator = Double.logGamma((nu + 1) / 2)
    let logDenominator = 0.5 * .log(nu * .pi) + Double.logGamma(nu / 2)
    let logKernel = -((nu + 1) / 2) * .log(1 + x * x / nu)
    let logValue = logNumerator - logDenominator + logKernel

    return logarithmic ? logValue : .exp(logValue)
  }

  public var mean: Double {
    return degreesOfFreedom > 1 ? .zero : .nan
  }

  public var variance: Double {
    if degreesOfFreedom > 2 {
      return degreesOfFreedom / (degreesOfFreedom - 2)
    } else if degreesOfFreedom > 1 {
      return .infinity
    } else {
      return .nan
    }
  }

  public var skewness: Double {
    return degreesOfFreedom > 3 ? .zero : .nan
  }

  public var excessKurtosis: Double {
    if degreesOfFreedom > 4 {
      return 6 / (degreesOfFreedom - 4)
    } else if degreesOfFreedom > 2 {
      return .infinity
    } else {
      return .nan
    }
  }

  public func cdf(x: Double, logarithmic: Bool = false) -> Double {
    let nu = degreesOfFreedom
    let z = nu / (nu + x * x)

    // When z is exactly 0 or 1 (extreme x values), handle edge cases directly.
    if z <= 0 {
      let result: Double = x < 0 ? 0 : 1
      return logarithmic ? (result == 0 ? -.infinity : 0) : result
    }
    if z >= 1 {
      let result = 0.5
      return logarithmic ? .log(result) : result
    }

    let ibeta = regularizedIncompleteBeta(x: z, alpha: nu / 2, beta: 0.5)

    let result: Double
    if x >= 0 {
      result = 1 - 0.5 * ibeta
    } else {
      result = 0.5 * ibeta
    }

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
  /// Sampling uses the ratio t = Z / sqrt(V / nu),
  /// where Z ~ N(0,1) and V ~ Chi-Squared(nu).
  public func sample() -> Double {
    let z = normal.sample()
    let v = chiSquared.sample()
    return z / (v / degreesOfFreedom).squareRoot()
  }

  /// Samples a specified number of values from the distribution.
  /// - parameter numberOfElements: The number of samples to generate.
  /// - returns: An array of sampled values.
  public func sample(_ numberOfElements: Int) -> [Double] {
    precondition(0 < numberOfElements, "The requested number of samples need to be greater than 0.")

    let zSamples = normal.sample(numberOfElements)
    let vSamples = chiSquared.sample(numberOfElements)
    return zip(zSamples, vSamples).map { z, v in
      z / (v / degreesOfFreedom).squareRoot()
    }
  }
}
