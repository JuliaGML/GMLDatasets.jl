using GMLDatasets
using GMLDatasets: read_idx, read_idx_split
using CodecZlib: GzipCompressorStream
using Test

# The reader is checked on files written here, so that the tests download nothing.
function idx_bytes(payload::AbstractVector{UInt8}, dims::Integer...; type_code = 0x08)
    io = IOBuffer()
    write(io, 0x00, 0x00, type_code, UInt8(length(dims)))
    for n in dims
        write(io, hton(UInt32(n)))
    end
    write(io, payload)
    take!(io)
end

# The data dependencies keep the files compressed, as the mirrors serve them.
function write_gz(file, bytes)
    stream = GzipCompressorStream(open(file, "w"))
    write(stream, bytes)
    close(stream)
end

@testset "read_idx" begin
    # Two images of three rows and two columns. The file stores each image row by row, so the first
    # index of the result runs along a row.
    payload = UInt8.(1:12)
    images = read_idx(IOBuffer(idx_bytes(payload, 2, 3, 2)))
    @test images isa Array{UInt8, 3}
    @test size(images) == (2, 3, 2)
    @test images[:, :, 1] == [1 3 5; 2 4 6]
    @test images[:, :, 2] == [7 9 11; 8 10 12]

    labels = read_idx(IOBuffer(idx_bytes(UInt8[7, 0, 9], 3)))
    @test labels isa Vector{UInt8}
    @test labels == [7, 0, 9]

    # A size above 255 needs every byte of the big-endian header.
    @test size(read_idx(IOBuffer(idx_bytes(zeros(UInt8, 300), 300)))) == (300,)

    @test_throws r"^ArgumentError: not an IDX file: the magic number does not start with two zero bytes$" read_idx(IOBuffer(vcat(
        0x01, idx_bytes(payload, 12)[2:end])))
    @test_throws r"^ArgumentError: unsupported IDX type code 0x0d; only unsigned bytes \(0x08\) are read$" read_idx(IOBuffer(idx_bytes(
        payload, 12; type_code = 0x0d)))
    @test_throws r"^ArgumentError: the IDX file holds fewer bytes than its header declares$" read_idx(IOBuffer(idx_bytes(
        payload[1:11], 12)))
    @test_throws r"^ArgumentError: the IDX file holds more bytes than its header declares$" read_idx(IOBuffer(idx_bytes(
        vcat(payload, 0x00), 12)))

    mktempdir() do dir
        bytes = idx_bytes(payload, 2, 3, 2)
        write(joinpath(dir, "images-idx3-ubyte"), bytes)
        write_gz(joinpath(dir, "images-idx3-ubyte.gz"), bytes)
        @test read_idx(joinpath(dir, "images-idx3-ubyte")) == images
        @test read_idx(joinpath(dir, "images-idx3-ubyte.gz")) == images

        write_gz(joinpath(dir, "short-idx1-ubyte.gz"), idx_bytes(payload[1:11], 12))
        @test_throws r"^ArgumentError: the IDX file holds fewer bytes than its header declares$" read_idx(joinpath(
            dir, "short-idx1-ubyte.gz"))
    end
end

@testset "read_idx_split" begin
    mktempdir() do dir
        pixels = UInt8[0, 51, 255, 128]
        write_gz(joinpath(dir, "t10k-images-idx3-ubyte.gz"), idx_bytes(pixels, 1, 2, 2))
        write_gz(joinpath(dir, "t10k-labels-idx1-ubyte.gz"), idx_bytes(UInt8[4], 1))

        data = read_idx_split(dir, :test)
        @test keys(data) == (:features, :targets)
        @test data.features isa Array{Float32, 3}
        @test size(data.features) == (2, 2, 1)
        @test vec(data.features) == pixels ./ Float32(255)
        @test data.features[3] === 1.0f0
        @test data.features[2] === 0.2f0
        @test data.targets isa Vector{Int}
        @test data.targets == [4]

        images, labels = data
        @test images === data.features
        @test labels === data.targets
    end
end

@testset "invalid split" begin
    @test_throws r"^ArgumentError: split must be :train or :test, got :validation$" mnist(:validation)
    @test_throws r"^ArgumentError: split must be :train or :test, got :validation$" fashion_mnist(:validation)
end
