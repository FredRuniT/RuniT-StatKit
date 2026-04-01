import Testing
import StatKit

@Suite("T-Test Tests", .tags(.hypothesisTesting))
struct TTestTests {

  // MARK: - One-sample t-test

  @Test("One-sample t-test with mean equal to hypothesized value")
  func oneSampleMeanEqualToHypothesized() async {
    // Symmetric data around 5.0
    let data = [4.0, 4.5, 5.0, 5.5, 6.0]
    let result = data.oneSampleTTest(variable: \.self, hypothesizedMean: 5.0)

    #expect(result.testStatistic.isApproximatelyEqual(to: 0.0, absoluteTolerance: 1e-10))
    #expect(result.pValue.isApproximatelyEqual(to: 1.0, absoluteTolerance: 1e-6))
    #expect(result.degreesOfFreedom.isApproximatelyEqual(to: 4.0, absoluteTolerance: 1e-10))
    #expect(!result.isSignificant)
  }

  @Test("One-sample t-test with mean far from hypothesized value")
  func oneSampleSignificant() async {
    // Mean is 100, hypothesized is 0 — should be highly significant
    let data = [98.0, 99.0, 100.0, 101.0, 102.0]
    let result = data.oneSampleTTest(variable: \.self, hypothesizedMean: 0.0)

    #expect(result.testStatistic > 50)
    #expect(result.pValue < 0.001)
    #expect(result.isSignificant)
  }

  @Test("One-sample t-test known values")
  func oneSampleKnownValues() async {
    // Data: [2, 4, 6, 8, 10], mean = 6, sd = sqrt(10), n = 5
    // t = (6 - 5) / (sqrt(10)/sqrt(5)) = 1 / sqrt(2) ≈ 0.7071
    // df = 4
    let data = [2.0, 4.0, 6.0, 8.0, 10.0]
    let result = data.oneSampleTTest(variable: \.self, hypothesizedMean: 5.0)

    let expectedT = 1.0 / 2.0.squareRoot()  // ≈ 0.7071
    #expect(result.testStatistic.isApproximatelyEqual(to: expectedT, absoluteTolerance: 1e-4))
    #expect(result.degreesOfFreedom.isApproximatelyEqual(to: 4.0, absoluteTolerance: 1e-10))
    // With t ≈ 0.707 and df = 4, p ≈ 0.519 (two-tailed)
    #expect(result.pValue.isApproximatelyEqual(to: 0.519, absoluteTolerance: 0.01))
    #expect(!result.isSignificant)
  }

  @Test("One-sample t-test with fewer than two elements returns NaN")
  func oneSampleInsufficientData() async {
    let empty = [Double]()
    let single = [5.0]

    let emptyResult = empty.oneSampleTTest(variable: \.self, hypothesizedMean: 0.0)
    #expect(emptyResult.testStatistic.isNaN)
    #expect(emptyResult.pValue.isNaN)
    #expect(emptyResult.degreesOfFreedom.isNaN)

    let singleResult = single.oneSampleTTest(variable: \.self, hypothesizedMean: 0.0)
    #expect(singleResult.testStatistic.isNaN)
    #expect(singleResult.pValue.isNaN)
  }

  @Test("One-sample t-test with all identical values equal to hypothesized mean")
  func oneSampleIdenticalValuesAtMean() async {
    let data = [5.0, 5.0, 5.0, 5.0]
    let result = data.oneSampleTTest(variable: \.self, hypothesizedMean: 5.0)

    #expect(result.testStatistic == 0)
    #expect(result.pValue == 1)
    #expect(!result.isSignificant)
  }

  @Test("One-sample t-test with all identical values not at hypothesized mean")
  func oneSampleIdenticalValuesNotAtMean() async {
    let data = [5.0, 5.0, 5.0, 5.0]
    let result = data.oneSampleTTest(variable: \.self, hypothesizedMean: 3.0)

    #expect(result.testStatistic.isInfinite)
    #expect(result.pValue == 0)
    #expect(result.isSignificant)
  }

  @Test("One-sample t-test with integer data")
  func oneSampleIntegerData() async {
    let data = [10, 12, 14, 16, 18]
    let result = data.oneSampleTTest(variable: \.self, hypothesizedMean: 14.0)

    #expect(result.testStatistic.isApproximatelyEqual(to: 0.0, absoluteTolerance: 1e-10))
    #expect(result.pValue.isApproximatelyEqual(to: 1.0, absoluteTolerance: 1e-6))
  }

  // MARK: - Two-sample t-test (Welch's)

  @Test("Two-sample t-test with identical distributions")
  func twoSampleIdenticalDistributions() async {
    let sample1 = [2.0, 4.0, 6.0, 8.0, 10.0]
    let sample2 = [2.0, 4.0, 6.0, 8.0, 10.0]

    let result = [Double].twoSampleTTest(sample1, sample2, variable: \.self)

    #expect(result.testStatistic.isApproximatelyEqual(to: 0.0, absoluteTolerance: 1e-10))
    #expect(result.pValue.isApproximatelyEqual(to: 1.0, absoluteTolerance: 1e-6))
    #expect(!result.isSignificant)
  }

  @Test("Two-sample t-test with clearly different means")
  func twoSampleDifferentMeans() async {
    let sample1 = [1.0, 2.0, 3.0, 4.0, 5.0]
    let sample2 = [100.0, 101.0, 102.0, 103.0, 104.0]

    let result = [Double].twoSampleTTest(sample1, sample2, variable: \.self)

    #expect(abs(result.testStatistic) > 10)
    #expect(result.pValue < 0.001)
    #expect(result.isSignificant)
  }

  @Test("Two-sample t-test known values")
  func twoSampleKnownValues() async {
    // Sample 1: [4, 5, 6, 7, 8], mean = 6, var = 2.5
    // Sample 2: [1, 2, 3, 4, 5], mean = 3, var = 2.5
    // t = (6 - 3) / sqrt(2.5/5 + 2.5/5) = 3 / sqrt(1.0) = 3.0
    // Welch df = (0.5 + 0.5)² / ((0.5²/4) + (0.5²/4)) = 1 / 0.125 = 8
    let sample1 = [4.0, 5.0, 6.0, 7.0, 8.0]
    let sample2 = [1.0, 2.0, 3.0, 4.0, 5.0]

    let result = [Double].twoSampleTTest(sample1, sample2, variable: \.self)

    #expect(result.testStatistic.isApproximatelyEqual(to: 3.0, absoluteTolerance: 1e-10))
    #expect(result.degreesOfFreedom.isApproximatelyEqual(to: 8.0, absoluteTolerance: 1e-10))
    // With t = 3 and df = 8, p ≈ 0.0170 (two-tailed)
    #expect(result.pValue.isApproximatelyEqual(to: 0.017, absoluteTolerance: 0.005))
    #expect(result.isSignificant)
  }

  @Test("Two-sample t-test with unequal sample sizes")
  func twoSampleUnequalSizes() async {
    let sample1 = [1.0, 2.0, 3.0]
    let sample2 = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]

    let result = [Double].twoSampleTTest(sample1, sample2, variable: \.self)

    #expect(abs(result.testStatistic) > 5)
    #expect(result.pValue < 0.01)
    #expect(result.isSignificant)
  }

  @Test("Two-sample t-test with insufficient data returns NaN")
  func twoSampleInsufficientData() async {
    let empty = [Double]()
    let single = [5.0]
    let valid = [1.0, 2.0, 3.0]

    let result1 = [Double].twoSampleTTest(empty, valid, variable: \.self)
    #expect(result1.testStatistic.isNaN)
    #expect(result1.pValue.isNaN)

    let result2 = [Double].twoSampleTTest(valid, single, variable: \.self)
    #expect(result2.testStatistic.isNaN)
    #expect(result2.pValue.isNaN)
  }

  @Test("Two-sample t-test with integer data")
  func twoSampleIntegerData() async {
    let sample1 = [10, 12, 14, 16, 18]
    let sample2 = [20, 22, 24, 26, 28]

    let result = [Int].twoSampleTTest(sample1, sample2, variable: \.self)

    #expect(abs(result.testStatistic) > 3)
    #expect(result.pValue < 0.01)
    #expect(result.isSignificant)
  }

  // MARK: - HypothesisTestResult

  @Test("HypothesisTestResult significance at different levels")
  func significanceLevels() async {
    // A p-value of 0.03 is significant at alpha = 0.05 but not at alpha = 0.01
    let resultDefault = HypothesisTestResult(testStatistic: 2.5, pValue: 0.03, degreesOfFreedom: 10)
    #expect(resultDefault.isSignificant)

    let resultStrict = HypothesisTestResult(testStatistic: 2.5, pValue: 0.03, degreesOfFreedom: 10, significanceLevel: 0.01)
    #expect(!resultStrict.isSignificant)
  }
}
