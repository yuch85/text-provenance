/** Per-metric score breakdown. */
export interface MetricScores {
  tokenContainment: number
  charNgramJaccard: number
  quoteBonus: number
  sentenceContainment: number
  lcsRatio: number
}

/** Weights for each metric. All five must be provided; they should sum to 1.0. */
export interface Weights {
  tokenContainment: number
  charNgramJaccard: number
  quoteBonus: number
  sentenceContainment: number
  lcsRatio: number
}

/** Options for the match() function. */
export interface MatchOptions {
  /** Metric weights. Defaults: tokenContainment=0.40, charNgramJaccard=0.50, quoteBonus=0.10, sentenceContainment=0.00, lcsRatio=0.00 */
  weights?: Partial<Weights>
  /** Minimum score threshold. Default: 0.04 */
  threshold?: number
  /** Maximum number of results to return. Default: 3 */
  topK?: number
}

/** A single match result with score and per-metric breakdown. */
export interface MatchResult {
  /** The matched candidate sentence. */
  sentence: string
  /** Index of this candidate in the original array. */
  index: number
  /** Combined weighted score. */
  score: number
  /** Individual metric scores. */
  metrics: MetricScores
}
