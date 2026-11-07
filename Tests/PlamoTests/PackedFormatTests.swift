import XCTest
import Metal
@testable import Plamo

final class PackedFormatTests: XCTestCase {
    func testTaggedBlockBounds() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal unavailable") }
        let path = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:path) }
        func archive(entries: [UInt32], bytes: Int, format: String = "plamo-bf16-pack-v2", base: Int = 107, tableBytes: Int? = nil) throws {
            let start = max(64,(entries.count*4+63)/64*64)
            let header: [String:Any] = ["format":format,"block":128,"minimum_base":base,
                "source_sha256":"7c9f0066d166c075bdbe83ada3d444e08bd7a6ee3f345f47a1f3c442aa6898c5",
                "payload_bytes":start+bytes,"tensors":["w":["shape":[1,entries.count*128],
                    "offsets":[0,tableBytes ?? entries.count*4],"data":[start,start+bytes]]]]
            let json = try JSONSerialization.data(withJSONObject:header)
            let length = (json.count+8+63)/64*64-8
            var count = UInt64(length).littleEndian
            var data = withUnsafeBytes(of:&count) { Data($0) }
            data.append(json); data.append(Data(repeating:32,count:length-json.count))
            entries.map(\.littleEndian).withUnsafeBytes { data.append(contentsOf:$0) }
            data.append(Data(repeating:0,count:start-entries.count*4+bytes))
            try data.write(to:path,options:.atomic)
        }
        for (entry,bytes) in [(UInt32(0),256),(UInt32(1),192),(UInt32(63),192)] {
            try archive(entries:[entry],bytes:bytes)
            let packed = try Packed(path:path.path,device:device)
            XCTAssertNotNil(try packed.tensor("w",[1,128]).packing)
        }
        for (entry,bytes) in [(UInt32(0),192),(UInt32(1),191),(UInt32(65),192),(UInt32.max,192)] {
            try archive(entries:[entry],bytes:bytes)
            let packed = try Packed(path:path.path,device:device)
            XCTAssertThrowsError(try packed.tensor("w",[1,128]))
        }
        try archive(entries:[1],bytes:192,tableBytes:3)
        XCTAssertThrowsError(try Packed(path:path.path,device:device).tensor("w",[1,128]))
        for (format,base) in [("plamo-bf16-pack-v1",107),("plamo-bf16-pack-v2",106)] {
            try archive(entries:[1],bytes:192,format:format,base:base)
            XCTAssertThrowsError(try Packed(path:path.path,device:device)) { error in
                XCTAssertTrue(String(describing:error).contains("scripts/pack_weights.py"))
            }
        }
        func check(_ entries: [UInt32], _ bytes: Int) throws {
            var end = 0, valid = true
            for entry in entries {
                let offset = Int(entry & ~UInt32(63)), size = entry & 63 == 0 ? 256 : 192
                if offset != end || offset > bytes-size { valid = false; break }
                end = offset+size
            }
            valid = valid && end == bytes
            try archive(entries:entries,bytes:bytes)
            try autoreleasepool {
                let packed = try Packed(path:path.path,device:device)
                if valid { XCTAssertNotNil(try packed.tensor("w",[1,entries.count*128]).packing) }
                else { XCTAssertThrowsError(try packed.tensor("w",[1,entries.count*128])) }
            }
        }
        for count in 0...19 {
            var entries = [UInt32](), end = 0
            for i in 0..<count {
                let tag: UInt32 = i%3 == 0 ? 0 : (i%3 == 1 ? 1 : 63)
                entries.append(UInt32(end)|tag); end += tag == 0 ? 256 : 192
            }
            try check(entries,end)
            for bytes in [0,max(0,end-1),end+1,end+64] { try check(entries,bytes) }
            for i in entries.indices {
                for change: UInt32 in [1,63,64,256,UInt32.max] {
                    var bad = entries; bad[i] ^= change
                    try check(bad,end)
                }
            }
        }
    }
}
