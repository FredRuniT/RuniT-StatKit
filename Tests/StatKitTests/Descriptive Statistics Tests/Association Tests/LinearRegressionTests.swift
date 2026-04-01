import Testing
import StatKit

@Suite("Linear Regression Tests", .tags(.linearRegression))
struct LinearRegressionTests {
  @Test("Perfect positive linear relationship")
  func perfectPositiveRelationship() async {
    let data = [
      SIMD2(x: 1.0, y: 3.0),
      SIMD2(x: 2.0, y: 5.0),
      SIMD2(x: 3.0, y: 7.0),
      SIMD2(x: 4.0, y: 9.0),
      SIMD2(x: 5.0, y: 11.0),
    ]

    let result = data.linearRegression(of: \.x, and: \.y)

    #expect(result.slope.isApproximatelyEqual(to: 2.0, absoluteTolerance: 1e-10))
    #expect(result.intercept.isApproximatelyEqual(to: 1.0, absoluteTolerance: 1e-10))
    #expect(result.rSquared.isApproximatelyEqual(to: 1.0, absoluteTolerance: 1e-10))
    #expect(result.residuals.allSatisfy { $0.isApproximatelyEqual(to: 0, absoluteTolerance: 1e-10) })
  }

  @Test("Perfect negative linear relationship")
  func perfectNegativeRelationship() async {
    let data = [
      SIMD2(x: 1.0, y: 10.0),
      SIMD2(x: 2.0, y: 7.0),
      SIMD2(x: 3.0, y: 4.0),
      SIMD2(x: 4.0, y: 1.0),
    ]

    let result = data.linearRegression(of: \.x, and: \.y)

    #expect(result.slope.isApproximatelyEqual(to: -3.0, absoluteTolerance: 1e-10))
    #expect(result.intercept.isApproximatelyEqual(to: 13.0, absoluteTolerance: 1e-10))
    #expect(result.rSquared.isApproximatelyEqual(to: 1.0, absoluteTolerance: 1e-10))
  }

  @Test("Noisy data yields expected regression values")
  func noisyData() async {
    // Known dataset: X = [1..10], Y = [10,20,27,30,35,38,49,56,62,69]
    let data = [
      SIMD2(x: 1.0, y: 10.0),
      SIMD2(x: 2.0, y: 20.0),
      SIMD2(x: 3.0, y: 27.0),
      SIMD2(x: 4.0, y: 30.0),
      SIMD2(x: 5.0, y: 35.0),
      SIMD2(x: 6.0, y: 38.0),
      SIMD2(x: 7.0, y: 49.0),
      SIMD2(x: 8.0, y: 56.0),
      SIMD2(x: 9.0, y: 62.0),
      SIMD2(x: 10.0, y: 69.0),
    ]

    let result = data.linearRegression(of: \.x, and: \.y)

    // slope = Σ(xi - x̄)(yi - ȳ) / Σ(xi - x̄)² = 515 / 82.5 ≈ 6.2424
    // intercept = ȳ - slope * x̄ = 39.6 - 6.2424 * 5.5 ≈ 5.2667
    #expect(result.slope.isApproximatelyEqual(to: 6.24242424, absoluteTolerance: 1e-4))
    #expect(result.intercept.isApproximatelyEqual(to: 5.26666667, absoluteTolerance: 1e-4))
    #expect(result.rSquared > 0.98)
    #expect(result.rSquared <= 1.0)
    #expect(result.standardError > 0)
    #expect(result.residuals.count == 10)
  }

  @Test("Prediction returns correct value")
  func prediction() async {
    let data = [
      SIMD2(x: 0.0, y: 5.0),
      SIMD2(x: 1.0, y: 7.0),
      SIMD2(x: 2.0, y: 9.0),
    ]

    let result = data.linearRegression(of: \.x, and: \.y)

    #expect(result.predict(3.0).isApproximatelyEqual(to: 11.0, absoluteTolerance: 1e-10))
    #expect(result.predict(0.0).isApproximatelyEqual(to: 5.0, absoluteTolerance: 1e-10))
  }

  @Test("Collection with fewer than two elements returns NaN")
  func insufficientData() async {
    let empty = [SIMD2<Double>]()
    let single = [SIMD2(x: 1.0, y: 2.0)]

    let emptyResult = empty.linearRegression(of: \.x, and: \.y)
    #expect(emptyResult.slope.isNaN)
    #expect(emptyResult.intercept.isNaN)
    #expect(emptyResult.rSquared.isNaN)
    #expect(emptyResult.standardError.isNaN)
    #expect(emptyResult.residuals.isEmpty)

    let singleResult = single.linearRegression(of: \.x, and: \.y)
    #expect(singleResult.slope.isNaN)
    #expect(singleResult.intercept.isNaN)
    #expect(singleResult.rSquared.isNaN)
    #expect(singleResult.standardError.isNaN)
  }

  @Test("Constant X values yield undefined slope")
  func constantXValues() async {
    let data = [
      SIMD2(x: 5.0, y: 1.0),
      SIMD2(x: 5.0, y: 2.0),
      SIMD2(x: 5.0, y: 3.0),
    ]

    let result = data.linearRegression(of: \.x, and: \.y)
    #expect(result.slope.isNaN)
    #expect(result.intercept.isNaN)
  }

  @Test("Integer data works through ConvertibleToReal")
  func integerData() async {
    let data = [
      SIMD2(x: 1, y: 2),
      SIMD2(x: 2, y: 4),
      SIMD2(x: 3, y: 6),
      SIMD2(x: 4, y: 8),
    ]

    let result = data.linearRegression(of: \.x, and: \.y)

    #expect(result.slope.isApproximatelyEqual(to: 2.0, absoluteTolerance: 1e-10))
    #expect(result.intercept.isApproximatelyEqual(to: 0.0, absoluteTolerance: 1e-10))
    #expect(result.rSquared.isApproximatelyEqual(to: 1.0, absoluteTolerance: 1e-10))
  }

  @Test("Exactly two elements produce valid regression with undefined standard error")
  func twoElements() async {
    let data = [
      SIMD2(x: 1.0, y: 3.0),
      SIMD2(x: 2.0, y: 5.0),
    ]

    let result = data.linearRegression(of: \.x, and: \.y)

    #expect(result.slope.isApproximatelyEqual(to: 2.0, absoluteTolerance: 1e-10))
    #expect(result.intercept.isApproximatelyEqual(to: 1.0, absoluteTolerance: 1e-10))
    #expect(result.rSquared.isApproximatelyEqual(to: 1.0, absoluteTolerance: 1e-10))
    // With n=2, df for SE = 0, so SE is undefined
    #expect(result.standardError.isNaN)
  }

  @Test("Residuals sum to approximately zero")
  func residualsSumToZero() async {
    let data = [
      SIMD2(x: 1.0, y: 2.3),
      SIMD2(x: 2.0, y: 4.1),
      SIMD2(x: 3.0, y: 5.8),
      SIMD2(x: 4.0, y: 8.2),
      SIMD2(x: 5.0, y: 9.9),
    ]

    let result = data.linearRegression(of: \.x, and: \.y)
    let residualSum = result.residuals.reduce(0, +)

    #expect(residualSum.isApproximatelyEqual(to: 0, absoluteTolerance: 1e-10))
  }
}
