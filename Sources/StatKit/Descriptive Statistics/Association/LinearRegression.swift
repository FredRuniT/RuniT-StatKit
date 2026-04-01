import RealModule

/// The result of fitting a simple linear regression model: Y = slope * X + intercept.
public struct LinearRegressionResult: Sendable {
  /// The slope of the regression line.
  public let slope: Double
  /// The y-intercept of the regression line.
  public let intercept: Double
  /// The coefficient of determination (R²), indicating the proportion of variance explained.
  public let rSquared: Double
  /// The standard error of the regression (residual standard error).
  public let standardError: Double
  /// The residuals (observed Y - predicted Y) for each observation.
  public let residuals: [Double]

  /// Predicts the Y value for a given X value using the fitted model.
  /// - parameter x: The X value for which to predict Y.
  /// - returns: The predicted Y value.
  @inlinable
  public func predict(_ x: Double) -> Double {
    slope * x + intercept
  }
}

public extension Collection {
  /// Fits a simple ordinary least squares linear regression: Y = slope * X + intercept.
  /// - parameter X: The independent variable.
  /// - parameter Y: The dependent variable.
  /// - returns: A `LinearRegressionResult` containing the slope, intercept, R², standard error, and residuals.
  ///
  /// Since linear regression requires at least two observations,
  /// this method returns NaN values for slope, intercept, R², and standard error
  /// if the collection contains fewer than two elements.
  /// The time complexity of this method is O(n).
  func linearRegression<T, U>(
    of X: KeyPath<Element, T>,
    and Y: KeyPath<Element, U>
  ) -> LinearRegressionResult
  where T: Comparable & Hashable & ConvertibleToReal,
        U: Comparable & Hashable & ConvertibleToReal
  {
    guard self.count > 1 else {
      return LinearRegressionResult(
        slope: .signalingNaN,
        intercept: .signalingNaN,
        rSquared: .signalingNaN,
        standardError: .signalingNaN,
        residuals: []
      )
    }

    let n = self.count.realValue
    let meanX = self.mean(variable: X)
    let meanY = self.mean(variable: Y)

    // Compute slope using OLS: slope = Σ((xi - x̄)(yi - ȳ)) / Σ((xi - x̄)²)
    var sumCrossDeviation: Double = 0
    var sumSquareDeviationX: Double = 0
    var sumSquareDeviationY: Double = 0

    for element in self {
      let dx = element[keyPath: X].realValue - meanX
      let dy = element[keyPath: Y].realValue - meanY
      sumCrossDeviation += dx * dy
      sumSquareDeviationX += dx * dx
      sumSquareDeviationY += dy * dy
    }

    // When all X values are identical, the slope is undefined.
    guard sumSquareDeviationX > 0 else {
      return LinearRegressionResult(
        slope: .signalingNaN,
        intercept: .signalingNaN,
        rSquared: .signalingNaN,
        standardError: .signalingNaN,
        residuals: self.map { element in
          element[keyPath: Y].realValue - meanY
        }
      )
    }

    let slope = sumCrossDeviation / sumSquareDeviationX
    let intercept = meanY - slope * meanX

    // Compute residuals and sum of squared residuals
    var ssRes: Double = 0
    let residuals: [Double] = self.map { element in
      let predicted = slope * element[keyPath: X].realValue + intercept
      let residual = element[keyPath: Y].realValue - predicted
      ssRes += residual * residual
      return residual
    }

    // R² = 1 - SS_res / SS_tot
    let ssTot = sumSquareDeviationY
    let rSquared: Double = ssTot > 0 ? 1 - ssRes / ssTot : (ssRes == 0 ? 1 : .signalingNaN)

    // Standard error = √(SS_res / (n - 2))
    let standardError: Double = n > 2 ? (ssRes / (n - 2)).squareRoot() : .signalingNaN

    return LinearRegressionResult(
      slope: slope,
      intercept: intercept,
      rSquared: rSquared,
      standardError: standardError,
      residuals: residuals
    )
  }
}
