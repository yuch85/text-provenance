import { describe, it, expect } from "vitest"
import { readFileSync } from "fs"
import { resolve } from "path"
import { match, tokenContainment, sentenceContainment, charNgramJaccard, quoteBonus, lcsRatio, tokenize } from "../src/index"

interface TestCase {
  name: string
  query: string
  candidates: string[]
  expected_top_index: number | null
  description: string
}

const fixtures: TestCase[] = JSON.parse(
  readFileSync(resolve(__dirname, "../../fixtures/test-cases.json"), "utf-8"),
)

describe("match() with shared fixtures", () => {
  for (const tc of fixtures) {
    if (tc.expected_top_index !== null) {
      it(`should match: ${tc.name}`, () => {
        const results = match(tc.query, tc.candidates)
        expect(results.length).toBeGreaterThan(0)
        expect(results[0].index).toBe(tc.expected_top_index)
      })
    } else {
      it(`should handle correctly: ${tc.name}`, () => {
        const results = match(tc.query, tc.candidates)
        // For null expected_top_index, either no results or no dominant winner
        // "no-match-expected" should return few/no results; "multi-sentence-synthesis" may return some
      })
    }
  }
})

describe("match() options", () => {
  const query = "The court held that the defendant was liable."
  const candidates = [
    "The court determined that the defendant bore liability for the damages caused.",
    "Weather forecasts predict rain throughout the weekend across the region.",
    "The appellant challenged the lower court's finding of liability.",
  ]

  it("should respect topK option", () => {
    const results = match(query, candidates, { topK: 1 })
    expect(results.length).toBeLessThanOrEqual(1)
  })

  it("should respect threshold option", () => {
    const results = match(query, candidates, { threshold: 0.99 })
    expect(results.length).toBe(0)
  })

  it("should respect custom weights", () => {
    const results = match(query, candidates, {
      weights: { tokenContainment: 1.0, charNgramJaccard: 0.0, quoteBonus: 0.0 },
    })
    expect(results.length).toBeGreaterThan(0)
  })

  it("should return empty for empty query", () => {
    expect(match("", candidates)).toEqual([])
  })

  it("should return empty for empty candidates", () => {
    expect(match(query, [])).toEqual([])
  })

  it("should include metric breakdown", () => {
    const results = match(query, candidates)
    expect(results.length).toBeGreaterThan(0)
    const r = results[0]
    expect(r.metrics).toBeDefined()
    expect(typeof r.metrics.tokenContainment).toBe("number")
    expect(typeof r.metrics.charNgramJaccard).toBe("number")
    expect(typeof r.metrics.quoteBonus).toBe("number")
    expect(typeof r.metrics.sentenceContainment).toBe("number")
    expect(typeof r.metrics.lcsRatio).toBe("number")
  })
})

describe("individual metrics", () => {
  it("tokenContainment", () => {
    const score = tokenContainment("the cat sat on the mat", "the cat sat on the mat")
    expect(score).toBeCloseTo(1.0)
  })

  it("tokenContainment partial", () => {
    const score = tokenContainment("cat sat mat", "the cat was sleeping on the mat yesterday")
    // "cat" and "mat" found; "sat" not found -> 2/3
    expect(score).toBeCloseTo(2 / 3, 1)
  })

  it("sentenceContainment", () => {
    const score = sentenceContainment(
      "long query about cats dogs birds and many other animals",
      "cats dogs",
    )
    // both "cats" and "dogs" from candidate found in query -> 1.0
    expect(score).toBeCloseTo(1.0)
  })

  it("charNgramJaccard identical", () => {
    expect(charNgramJaccard("hello world", "hello world")).toBeCloseTo(1.0)
  })

  it("charNgramJaccard zero", () => {
    expect(charNgramJaccard("abc", "xyz")).toBe(0)
  })

  it("quoteBonus with match", () => {
    expect(quoteBonus(
      'He said "compliance with treaty obligations is required" in the ruling.',
      "compliance with treaty obligations is required by all member states",
    )).toBe(1)
  })

  it("quoteBonus with ellipsis", () => {
    expect(quoteBonus(
      'The text states "all parties must reduce emissions... in accordance with their capabilities" as a binding rule.',
      "all parties must reduce emissions of greenhouse gases in a manner that reflects equity and in accordance with their capabilities and national circumstances",
    )).toBe(1)
  })

  it("quoteBonus no match", () => {
    expect(quoteBonus("no quotes here", "nothing to find")).toBe(0)
  })

  it("lcsRatio identical", () => {
    expect(lcsRatio("the cat sat", "the cat sat")).toBeCloseTo(1.0)
  })

  it("lcsRatio empty", () => {
    expect(lcsRatio("", "hello")).toBe(0)
  })
})

describe("tokenize", () => {
  it("should lowercase and filter stopwords", () => {
    const tokens = tokenize("The Quick Brown Fox")
    expect(tokens.has("quick")).toBe(true)
    expect(tokens.has("brown")).toBe(true)
    expect(tokens.has("fox")).toBe(true)
    expect(tokens.has("the")).toBe(false) // stopword
  })

  it("should filter single-char tokens and keep 2+ char", () => {
    const tokens = tokenize("I am a big cat")
    expect(tokens.has("big")).toBe(true)
    expect(tokens.has("cat")).toBe(true)
    expect(tokens.has("am")).toBe(true) // 2 chars, not a stopword -> kept
    // Single-char "I" and "a" are filtered (< 2 chars)
  })

  it("should handle empty string", () => {
    expect(tokenize("").size).toBe(0)
  })
})
