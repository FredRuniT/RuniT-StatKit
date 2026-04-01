/// The result of a hypothesis test.
public struct HypothesisTestResult: Sendable {
  /// The computed test statistic.
  public let testStatistic: Double
  /// The p-value associated with the test statistic.
  public let pValue: Double
  /// The degrees of freedom for the test.
  public let degreesOfFreedom: Double
  /// Whether the result is statistically significant at the given significance level.
  public let isSignificant: Bool

  /// Creates a new hypothesis test result.
  /// - parameter testStatistic: The computed test statistic.
  /// - parameter pValue: The p-value associated with the test statistic.
  /// - parameter degreesOfFreedom: The degrees of freedom for the test.
  /// - parameter significanceLevel: The threshold for significance (default 0.05).
  public init(
    testStatistic: Double,
    pValue: Double,
    degreesOfFreedom: Double,
    significanceLevel: Double = 0.05
  ) {
    self.testStatistic = testStatistic
    self.pValue = pValue
    self.degreesOfFreedom = degreesOfFreedom
    self.isSignificant = pValue < significanceLevel
  }
}
