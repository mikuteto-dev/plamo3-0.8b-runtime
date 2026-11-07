// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "Plamo",
    platforms: [.macOS(.v14)],
    products: [.executable(name: "plamo", targets: ["Plamo"])],
    targets: [
        .executableTarget(name: "Plamo", resources: [.copy("Kernels.metal"), .copy("Batch.metal")]),
        .testTarget(name: "PlamoTests", dependencies: ["Plamo"])
    ]
)
