import Foundation

// Optional decoding controls applied on top of greedy argmax.
// The reference implementation and the verification path stay pure greedy; these
// controls only run when the caller opts in, so benchmark parity is preserved.
enum Sampler {
    // Chooses the next token from `logits`, which is modified in place.
    // `history` is the token sequence already in context (prompt plus generated tokens).
    static func select(_ logits: inout [Float], history: [Int], penalty: Float, noRepeatNGram: Int,
                       frequency: Float = 0, presence: Float = 0) -> Int {
        if penalty != 1 || presence != 0 || frequency != 0 {
            var counts = [Int: Float]()
            for id in history where (0..<logits.count).contains(id) { counts[id, default: 0] += 1 }
            for (id, count) in counts {
                if penalty != 1 { logits[id] = logits[id] > 0 ? logits[id] / penalty : logits[id] * penalty }
                if presence != 0 { logits[id] -= presence }
                if frequency != 0 { logits[id] -= frequency * count }
            }
        }
        if noRepeatNGram > 1 {
            for id in forbidden(history, noRepeatNGram) where (0..<logits.count).contains(id) {
                logits[id] = -Float.infinity
            }
        }
        var best = -Float.infinity
        var index = 0
        for (i, value) in logits.enumerated() where value > best {
            best = value
            index = i
        }
        return index
    }

    // Returns the shortest period (1...32) if the tail of `tokens` is a block repeated
    // at least `repeats` times and spanning at least 24 tokens; otherwise nil.
    static func repetitionPeriod(_ tokens: [Int], repeats: Int) -> Int? {
        guard repeats >= 2, tokens.count >= repeats else { return nil }
        for period in 1...32 {
            let count = max(repeats, (24 + period - 1) / period)
            let span = period * count
            guard tokens.count >= span else { continue }
            let start = tokens.count - span
            var periodic = true
            for offset in 0..<span where tokens[start+offset] != tokens[start+offset % period] {
                periodic = false
                break
            }
            if periodic { return period }
        }
        return nil
    }

    // Token IDs that would complete an n-gram already present in `history`.
    static func forbidden(_ history: [Int], _ n: Int) -> Set<Int> {
        guard n > 1, history.count >= n else { return [] }
        let span = n - 1
        let last = history.count - span
        var result = Set<Int>()
        var start = 0
        while start < last {
            var matches = true
            for offset in 0..<span where history[start+offset] != history[last+offset] {
                matches = false
                break
            }
            if matches { result.insert(history[start+span]) }
            start += 1
        }
        return result
    }
}
