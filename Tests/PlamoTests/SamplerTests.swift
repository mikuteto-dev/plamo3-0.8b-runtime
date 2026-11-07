import XCTest
@testable import Plamo

final class SamplerTests: XCTestCase {
    func testGreedyPicksFirstMaximum() {
        var logits: [Float] = [1, 2, 2, 0]
        XCTAssertEqual(Sampler.select(&logits, history: [], penalty: 1, noRepeatNGram: 0), 1)
    }

    func testRepetitionPenaltyMovesChoice() {
        var positive: [Float] = [1, 5, 4]
        XCTAssertEqual(Sampler.select(&positive, history: [1], penalty: 2, noRepeatNGram: 0), 2)
        var negative: [Float] = [-1, -5, -4]
        XCTAssertEqual(Sampler.select(&negative, history: [1], penalty: 2, noRepeatNGram: 0), 0)
    }

    func testFrequencyPenaltyCompoundsWithCount() {
        var single: [Float] = [0, 5, 4]
        XCTAssertEqual(Sampler.select(&single, history: [1], penalty: 1, noRepeatNGram: 0, frequency: 0.5), 1)
        var repeated: [Float] = [0, 5, 4]
        XCTAssertEqual(Sampler.select(&repeated, history: [1, 1, 1], penalty: 1, noRepeatNGram: 0, frequency: 0.5), 2)
    }

    func testForbiddenNgramsBlockTheCompletion() {
        // The last bigram [2, 3] occurred before and was followed by 5, so 5 is blocked.
        XCTAssertEqual(Sampler.forbidden([0, 2, 3, 5, 2, 3], 3), Set([5]))
        var logits: [Float] = [0, 0, 0, 0, 0, 9]
        XCTAssertEqual(Sampler.select(&logits, history: [0, 2, 3, 5, 2, 3], penalty: 1, noRepeatNGram: 3), 0)
    }

    func testRepetitionPeriodDetectsRepeatedBlocks() {
        XCTAssertEqual(Sampler.repetitionPeriod([Int](repeating: 7, count: 24), repeats: 8), 1)
        let block = [1, 2, 3, 4]
        let repeated = Array(repeating: block, count: 8).flatMap { $0 }
        XCTAssertEqual(Sampler.repetitionPeriod(repeated, repeats: 8), 4)
        XCTAssertNil(Sampler.repetitionPeriod([1, 2, 3, 4, 5, 6, 7, 8], repeats: 8))
        XCTAssertNil(Sampler.repetitionPeriod([1, 2, 3, 4], repeats: 8))
    }
}
