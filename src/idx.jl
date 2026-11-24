@doc raw"""
    read_idx(io::IO)
    read_idx(file::AbstractString)

Read an IDX file of unsigned bytes, the format MNIST and Fashion-MNIST are distributed in, and return
its payload as an `Array{UInt8}`.

The header is a magic number — two zero bytes, the type code `0x08` for unsigned bytes and the
number of dimensions ``d`` — followed by ``d`` big-endian `UInt32` sizes ``n_1, \ldots, n_d``. The
payload is stored with the last index running fastest, so it is returned with its dimensions
reversed, ``n_d \times \cdots \times n_1``: an image file of ``n`` images of ``r`` rows and ``c``
columns gives a ``c \times r \times n`` array, and a label file a vector of length ``n``.

A `file` whose name ends in `.gz` is decompressed while it is read.
"""
function read_idx(io::IO)
    magic = read(io, 4)
    length(magic) == 4 || throw(EOFError())
    (magic[1] == 0x00 && magic[2] == 0x00) ||
        throw(ArgumentError("not an IDX file: the magic number does not start with two zero bytes"))
    magic[3] == 0x08 ||
        throw(ArgumentError("unsupported IDX type code $(repr(magic[3])); only unsigned bytes (0x08) are read"))
    dims = ntuple(_ -> Int(ntoh(read(io, UInt32))), Int(magic[4]))
    payload = read(io, prod(dims))
    length(payload) == prod(dims) ||
        throw(ArgumentError("the IDX file holds fewer bytes than its header declares"))
    eof(io) ||
        throw(ArgumentError("the IDX file holds more bytes than its header declares"))
    reshape(payload, reverse(dims))
end

function read_idx(file::AbstractString)
    endswith(file, ".gz") || return open(read_idx, file)
    stream = GzipDecompressorStream(open(file))
    try
        read_idx(stream)
    finally
        close(stream)
    end
end

const MNIST_DATADEP = "GMLDatasets_MNIST"
const FASHION_MNIST_DATADEP = "GMLDatasets_FashionMNIST"

const IDX_FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]

# The mirrors and checksums are those `MLDatasets` 0.7 registers for its `MNIST` and `FashionMNIST`
# data dependencies, and the files stay compressed, as they do there. The names differ from its
# `"MNIST"` and `"FashionMNIST"`, so that a session that loads both packages registers each name once.
function register_datadeps()
    register(DataDep(MNIST_DATADEP,
        """
        Dataset: THE MNIST DATABASE of handwritten digits
        Authors: Yann LeCun, Corinna Cortes, Christopher J.C. Burges
        Website: http://yann.lecun.com/exdb/mnist/

        [LeCun et al., 1998a]
            Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner.
            "Gradient-based learning applied to document recognition."
            Proceedings of the IEEE, 86(11):2278-2324, November 1998

        The authors of MNIST do not state terms of use; read the website to make sure you want to
        download the data set. Using the data responsibly and respecting copyright remains your
        responsibility.
        """,
        "https://ossci-datasets.s3.amazonaws.com/mnist/" .* IDX_FILES,
        "0bb1d5775d852fc5bb32c76ca15a7eb4e9a3b1514a2493f7edfcf49b639d7975"))
    register(DataDep(FASHION_MNIST_DATADEP,
        """
        Dataset: Fashion-MNIST
        Authors: Han Xiao, Kashif Rasul, Roland Vollgraf
        Website: https://github.com/zalandoresearch/fashion-mnist
        License: MIT

        [Han Xiao et al. 2017]
            Han Xiao, Kashif Rasul, and Roland Vollgraf.
            "Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms."
            arXiv:1708.07747

        Using the data responsibly and respecting copyright remains your responsibility.
        """,
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/" .* IDX_FILES,
        "c916b6e00d3083643332b70f3c5c3543d3941334b802e252976893969ee6af67"))
end

function idx_dataset(depname::AbstractString, split::Symbol)
    split in (:train, :test) ||
        throw(ArgumentError("split must be :train or :test, got :$split"))
    read_idx_split(@datadep_str(depname), split)
end

# The images are scaled to [0, 1] in `Float32` and the labels widened to `Int`, which is what
# `MLDatasets.MNIST(split = split)[:]` returns.
function read_idx_split(dir::AbstractString, split::Symbol)
    prefix = split === :train ? "train" : "t10k"
    images = read_idx(joinpath(dir, "$prefix-images-idx3-ubyte.gz"))::Array{UInt8, 3}
    labels = read_idx(joinpath(dir, "$prefix-labels-idx1-ubyte.gz"))::Vector{UInt8}
    (features = images ./ Float32(255), targets = Vector{Int}(labels))
end
