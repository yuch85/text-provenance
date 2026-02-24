import type { MatchOptions, MatchResult, Weights, MetricScores } from "./types"
import { tokenContainment, sentenceContainment, charNgramJaccard, quoteBonus, lcsRatio } from "./metrics"

const DEFAULT_WEIGHTS: Weights = {
  tokenContainment: 0.40,
  charNgramJaccard: 0.50,
  quoteBonus: 0.10,
  sentenceContainment: 0.00,
  lcsRatio: 0.00,
}

const DEFAULT_THRESHOLD = 0.04
const DEFAULT_TOP_K = 3

/**
 * Find the candidate sentences that best match a query text.
 *
 * Scores each candidate using a weighted combination of lightweight text
 * similarity metrics and returns the top matches above a threshold.
 *
 * @param query - The text to trace back to its source.
 * @param candidates - Array of candidate source sentences.
 * @param options - Optional weights, threshold, and topK.
 * @returns Ranked match results with scores and per-metric breakdowns.
 */
export function match(query: string, candidates: string[], options?: MatchOptions): MatchResult[] {
  if (!query || candidates.length === 0) return []

  const weights: Weights = { ...DEFAULT_WEIGHTS, ...options?.weights }
  const threshold = options?.threshold ?? DEFAULT_THRESHOLD
  const topK = options?.topK ?? DEFAULT_TOP_K

  const scored: MatchResult[] = candidates.map((candidate, index) => {
    const metrics: MetricScores = {
      tokenContainment: tokenContainment(query, candidate),
      charNgramJaccard: charNgramJaccard(query, candidate),
      quoteBonus: quoteBonus(query, candidate),
      sentenceContainment: sentenceContainment(query, candidate),
      lcsRatio: lcsRatio(query, candidate),
    }

    const score =
      weights.tokenContainment * metrics.tokenContainment +
      weights.charNgramJaccard * metrics.charNgramJaccard +
      weights.quoteBonus * metrics.quoteBonus +
      weights.sentenceContainment * metrics.sentenceContainment +
      weights.lcsRatio * metrics.lcsRatio

    return { sentence: candidate, index, score, metrics }
  })

  scored.sort((a, b) => b.score - a.score)

  return scored.slice(0, topK).filter((r) => r.score >= threshold)
}
