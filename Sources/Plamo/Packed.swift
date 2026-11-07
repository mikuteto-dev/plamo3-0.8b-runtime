import Foundation
import Metal
import Darwin

final class Packed {
    let buffer: MTLBuffer
    private let mapping: Mapping
    private let tensors: [String: [String: Any]]
    private let dataStart: Int
    private let payload: Int

    init(path: String, device: MTLDevice) throws {
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw Failure.message("Packed weights missing; run scripts/pack_weights.py first") }
        defer { close(fd) }
        var info = stat()
        guard fstat(fd, &info) == 0, info.st_size > 8, info.st_size <= 2_000_000_000 else {
            throw Failure.message("Invalid packed file size")
        }
        let size = Int(info.st_size), page = Int(getpagesize())
        let length = (size+page-1)/page*page
        guard let pointer = mmap(nil, length, PROT_READ, MAP_PRIVATE, fd, 0), pointer != MAP_FAILED else {
            throw Failure.message("Cannot map packed weights")
        }
        let mapping = Mapping(pointer, length)
        self.mapping = mapping
        let count = pointer.loadUnaligned(as: UInt64.self).littleEndian
        guard count <= 1_048_576, count < size-8, (count+8)%64 == 0 else {
            throw Failure.message("Invalid packed header")
        }
        dataStart = Int(count)+8
        guard let header = try JSONSerialization.jsonObject(with: Data(bytes: pointer+8, count: Int(count))) as? [String: Any],
              header["format"] as? String == "plamo-bf16-pack-v2", header["block"] as? Int == 128,
              header["minimum_base"] as? Int == 107,
              header["source_sha256"] as? String == "7c9f0066d166c075bdbe83ada3d444e08bd7a6ee3f345f47a1f3c442aa6898c5",
              let bytes = header["payload_bytes"] as? Int, bytes == size-dataStart,
              let tensors = header["tensors"] as? [String: [String: Any]] else {
            throw Failure.message("Incompatible packed archive; run scripts/pack_weights.py to regenerate")
        }
        self.tensors = tensors
        payload = bytes
        guard let b = device.makeBuffer(bytesNoCopy: pointer, length: length, options: .storageModeShared,
                                       deallocator: { [mapping] _, _ in withExtendedLifetime(mapping) {} }) else {
            throw Failure.message("Cannot share packed weights with Metal")
        }
        buffer = b
    }

    func tensor(_ name: String, _ shape: [Int]) throws -> Ref {
        guard let record = tensors[name], record["shape"] as? [Int] == shape else {
            throw Failure.message("Packed tensor shape mismatch: \(name)")
        }
        func region(_ key: String) throws -> (start: Int, size: Int) {
            guard let r = record[key] as? [Int], r.count == 2, r[0] >= 0, r[0]%64 == 0,
                  r[1] >= r[0], r[1] <= payload else { throw Failure.message("Invalid packed region: \(name).\(key)") }
            return (dataStart+r[0], r[1]-r[0])
        }
        if shape.count == 1 {
            let raw = try region("raw")
            guard raw.size == shape[0]*2 else { throw Failure.message("Invalid packed norm") }
            return Ref(buffer: buffer, offset: raw.start)
        }
        let count = shape.reduce(1, *)/128
        let offsets = try region("offsets"), data = try region("data")
        guard offsets.size == count*4 else { throw Failure.message("Invalid packed block table") }
        let table = (mapping.pointer+offsets.start).assumingMemoryBound(to: UInt32.self)
        var end = 0, i = 0
        while i+4<=count {
            let raw = UnsafeRawPointer(table+i).loadUnaligned(as:SIMD4<UInt32>.self)
            let entry = SIMD4(raw[0].littleEndian,raw[1].littleEndian,raw[2].littleEndian,raw[3].littleEndian)
            let offsets = entry & SIMD4(repeating:~UInt32(63))
            let sizes = SIMD4<UInt32>(repeating:192).replacing(with:256,where:(entry & SIMD4(repeating:63)) .== SIMD4(repeating:0))
            // A valid chain starts below 2 GB, so four steps cannot wrap UInt32.
            let ends = offsets &+ sizes
            guard offsets == SIMD4(UInt32(end),ends[0],ends[1],ends[2]), Int(ends[3])<=data.size else {
                throw Failure.message("Packed block out of range: \(name)")
            }
            end = Int(ends[3]); i += 4
        }
        for j in i..<count {
            let entry = table[j].littleEndian
            let offset = Int(entry & ~UInt32(63))
            let bytes = entry & 63 == 0 ? 256 : 192
            guard offset == end, offset <= data.size-bytes else {
                throw Failure.message("Packed block out of range: \(name)")
            }
            end = offset+bytes
        }
        guard end == data.size else { throw Failure.message("Packed payload size mismatch") }
        return Ref(buffer: buffer, offset: data.start, packing: Packing(offsets: offsets.start))
    }
}
