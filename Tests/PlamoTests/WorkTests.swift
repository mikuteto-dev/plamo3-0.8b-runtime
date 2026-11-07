import XCTest
@testable import Plamo

final class WorkTests: XCTestCase {
    func testScratchShrinksWithoutChangingOutput() throws {
        guard FileManager.default.fileExists(atPath:"models/plamo/model.p12") else { throw XCTSkip("Pack weights first") }
        let compact=try Runtime(directory:"models/plamo",context:4096,packed:true)
        let retained=try Runtime(directory:"models/plamo",context:4096,packed:true)
        let prefix=[1]+Array(500..<515)
        for model in [compact,retained] {
            try model.forward(prefix)
            XCTAssertTrue(model.rotatedKeys.buffer === model.activated.buffer)
            XCTAssertTrue(model.rotatedKeys.buffer === model.normalized.buffer)
            XCTAssertEqual(model.activated.buffer.length,2064*128*2)
        }
        XCTAssertEqual(try compact.prefill([600]),try retained.forward([600]))
        XCTAssertEqual(compact.activated.buffer.length,6144*4)
        XCTAssertEqual(retained.activated.buffer.length,2064*128*2)
        XCTAssertEqual(compact.logitValues().map(\.bitPattern),retained.logitValues().map(\.bitPattern))
        XCTAssertEqual(try compact.forward([601]),try retained.forward([601]))
        XCTAssertEqual(compact.logitValues().map(\.bitPattern),retained.logitValues().map(\.bitPattern))
        let position=compact.position,logits=compact.logitValues().map(\.bitPattern)
        XCTAssertThrowsError(try compact.forward([107520]))
        XCTAssertEqual(compact.position,position)
        XCTAssertEqual(compact.activated.buffer.length,6144*4)
        XCTAssertEqual(compact.logitValues().map(\.bitPattern),logits)
    }

    func testRotaryModeGrowsAndShrinksScratch() throws {
        guard FileManager.default.fileExists(atPath:"models/plamo/model.p12") else { throw XCTSkip("Pack weights first") }
        let compact=try Runtime(directory:"models/plamo",context:4096,packed:true)
        let retained=try Runtime(directory:"models/plamo",context:4096,packed:true)
        retained.fuseRotary=false
        let prefix=[1]+(0..<128).map { 500+$0%37 }
        _ = try compact.prefill(prefix)
        for first in stride(from:0,to:prefix.count,by:16) {
            try retained.forward(Array(prefix[first..<min(first+16,prefix.count)]))
        }
        XCTAssertEqual(compact.activated.buffer.length,6144*4)
        XCTAssertEqual(compact.logitValues().map(\.bitPattern),retained.logitValues().map(\.bitPattern))
        compact.fuseRotary=false
        XCTAssertEqual(try compact.forward([601]),try retained.forward([601]))
        XCTAssertEqual(compact.activated.buffer.length,2064*128*2)
        XCTAssertEqual(compact.logitValues().map(\.bitPattern),retained.logitValues().map(\.bitPattern))
        compact.fuseRotary=true
        XCTAssertEqual(try compact.prefill([602]),try retained.forward([602]))
        XCTAssertEqual(compact.activated.buffer.length,6144*4)
        XCTAssertEqual(compact.logitValues().map(\.bitPattern),retained.logitValues().map(\.bitPattern))
    }

    func testVerificationUsesOnlyActivationScratch() throws {
        guard FileManager.default.fileExists(atPath:"models/plamo/auxiliary.safetensors") else { throw XCTSkip("Train adapters first") }
        let model=try Runtime(directory:"models/plamo",context:4096,packed:true,mtp:true)
        _ = try model.prefill([1]+Array(500..<532))
        XCTAssertEqual(model.activated.buffer.length,2*6144*4)
        try model.forward([600,700],all:true,sequential:true)
        XCTAssertEqual(model.activated.buffer.length,2*6144*4)
        XCTAssertEqual(model.lastOutputRows,2)
        try model.rewind(model.position-1,hiddenIndex:0)
        XCTAssertEqual(model.activated.buffer.length,2*6144*4)
        XCTAssertEqual(model.lastOutputRows,1)
        XCTAssertEqual(try model.draft(800).count,1)
    }
}
