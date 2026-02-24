/** Default English stopwords (47 common words). */
export const DEFAULT_STOPWORDS = new Set([
  "the", "and", "that", "this", "with", "from", "which", "have", "has",
  "had", "been", "were", "was", "are", "for", "not", "but", "its",
  "also", "would", "could", "should", "may", "can", "will", "shall",
  "does", "did", "into", "than", "such", "each", "any", "all", "own",
  "other", "more", "most", "very", "only", "just", "about", "between",
  "through", "under", "over", "after", "before", "when", "where", "while",
  "both", "being", "their", "there", "these", "those", "they", "them",
])

/**
 * Tokenize text into a set of lowercase alphanumeric strings (>= 2 chars),
 * with stopword removal.
 *
 * @param text - Input text to tokenize.
 * @param stopwords - Set of words to filter out. Defaults to DEFAULT_STOPWORDS.
 * @returns Set of filtered tokens.
 */
export function tokenize(text: string, stopwords: Set<string> = DEFAULT_STOPWORDS): Set<string> {
  const tokens = text.toLowerCase().match(/[a-z0-9\u00C0-\u024F]{2,}/g)
  if (!tokens) return new Set()
  return new Set(tokens.filter((t) => !stopwords.has(t)))
}
