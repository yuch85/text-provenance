import { tokenize, DEFAULT_STOPWORDS } from "./tokenizer"

/**
 * Token containment: fraction of query's tokens found in candidate.
 * Measures how well the candidate "explains" the query.
 */
export function tokenContainment(query: string, candidate: string, stopwords: Set<string> = DEFAULT_STOPWORDS): number {
  const queryTokens = tokenize(query, stopwords)
  const candidateTokens = tokenize(candidate, stopwords)
  if (queryTokens.size === 0) return 0
  let hits = 0
  for (const t of queryTokens) {
    if (candidateTokens.has(t)) hits++
  }
  return hits / queryTokens.size
}

/**
 * Sentence containment: fraction of candidate's tokens found in query.
 * The reverse direction of tokenContainment.
 */
export function sentenceContainment(query: string, candidate: string, stopwords: Set<string> = DEFAULT_STOPWORDS): number {
  const queryTokens = tokenize(query, stopwords)
  const candidateTokens = tokenize(candidate, stopwords)
  if (candidateTokens.size === 0) return 0
  let hits = 0
  for (const t of candidateTokens) {
    if (queryTokens.has(t)) hits++
  }
  return hits / candidateTokens.size
}

/**
 * Character n-gram Jaccard similarity.
 * |ngrams(a) ∩ ngrams(b)| / |ngrams(a) ∪ ngrams(b)|
 *
 * Captures morphological overlap (e.g., "criminality" vs "criminal").
 *
 * @param a - First text.
 * @param b - Second text.
 * @param n - N-gram size. Default: 4.
 */
export function charNgramJaccard(a: string, b: string, n: number = 4): number {
  const la = a.toLowerCase()
  const lb = b.toLowerCase()
  if (la.length < n && lb.length < n) return 0

  const setA = new Set<string>()
  for (let i = 0; i <= la.length - n; i++) setA.add(la.slice(i, i + n))

  const setB = new Set<string>()
  for (let i = 0; i <= lb.length - n; i++) setB.add(lb.slice(i, i + n))

  let inter = 0
  for (const g of setA) if (setB.has(g)) inter++
  const union = setA.size + setB.size - inter

  return union > 0 ? inter / union : 0
}

/**
 * Quote bonus: 1.0 if candidate contains a quoted substring from query
 * of at least minLength characters, else 0.0.
 *
 * Handles ellipsis in quotes: "text A... text B" splits on ellipsis and
 * checks if all fragments (>= minLength) appear in the candidate.
 *
 * @param query - Text that may contain quoted strings.
 * @param candidate - Text to search for quoted substrings.
 * @param minLength - Minimum quote length. Default: 15.
 */
export function quoteBonus(query: string, candidate: string, minLength: number = 15): number {
  const quotes = extractQuotes(query, minLength)
  if (quotes.length === 0) return 0
  const lower = candidate.toLowerCase()
  for (const q of quotes) {
    const ql = q.toLowerCase()
    if (lower.includes(ql)) return 1
    // Handle ellipsis: "text A... text B"
    if (ql.includes("...") || ql.includes("\u2026")) {
      const fragments = ql.split(/\.{3}|\u2026/).map((f) => f.trim()).filter((f) => f.length >= minLength)
      if (fragments.length >= 2 && fragments.every((f) => lower.includes(f))) return 1
    }
  }
  return 0
}

/** Extract quoted strings (>= minLength chars) from text. */
function extractQuotes(text: string, minLength: number): string[] {
  const quotes: string[] = []
  const re = new RegExp(`["'\\u201C\\u201D\\u2018\\u2019]([^"'\\u201C\\u201D\\u2018\\u2019]{${minLength},}?)["'\\u201C\\u201D\\u2018\\u2019]`, "g")
  let m: RegExpExecArray | null
  while ((m = re.exec(text)) !== null) {
    quotes.push(m[1].trim())
  }
  return quotes
}

/**
 * Word-level Longest Common Subsequence ratio.
 * LCS length / max(|wordsA|, |wordsB|).
 * Uses space-optimized two-row DP.
 */
export function lcsRatio(a: string, b: string): number {
  if (a.length === 0 || b.length === 0) return 0

  const wa = a.toLowerCase().split(/\s+/)
  const wb = b.toLowerCase().split(/\s+/)

  const m = wa.length
  const n = wb.length

  let prev = new Uint16Array(n + 1)
  let curr = new Uint16Array(n + 1)

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (wa[i - 1] === wb[j - 1]) {
        curr[j] = prev[j - 1] + 1
      } else {
        curr[j] = Math.max(prev[j], curr[j - 1])
      }
    }
    ;[prev, curr] = [curr, prev]
    curr.fill(0)
  }

  const lcsLen = prev[n]
  return lcsLen / Math.max(m, n)
}
