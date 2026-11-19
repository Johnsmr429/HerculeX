"""How the Apple lane builds kernels: clang for the M1 and later, into dynamic libraries."""

C_FLAGS = ("-O3", "-mcpu=apple-m1")
SUFFIX = "dylib"


def key_words(flags, libraries):
    """What decides a build besides its sources."""
    return (*flags, *libraries)


def command(flags, out, sources, libraries):
    """The compiler call that builds sources into out."""
    return ["clang", *flags, "-dynamiclib", "-o", str(out), *map(str, sources), *libraries]
