"""Native build configuration kept small enough to audit."""
from __future__ import annotations

import os
import sys

from setuptools import Extension, setup


compile_args = (["/O2", "/DNDEBUG"] if sys.platform == "win32"
                else ["-O3", "-DNDEBUG", "-std=c11"])
link_args: list[str] = []
libraries = ["m"] if sys.platform != "win32" else []

# Portable wheels must not use -march=native. OpenMP is opt-in because many
# macOS/Windows toolchains do not ship a compatible runtime by default.
if os.environ.get("MONOTONIC_ORDER_OPENMP") == "1":
    if sys.platform == "win32":
        compile_args.append("/openmp")
    else:
        compile_args.append("-fopenmp")
        link_args.append("-fopenmp")

setup(
    ext_modules=[
        Extension(
            "monotonic_order._native",
            sources=["src/monotonic_order/_native_module.c", "monotonic_order.c"],
            include_dirs=["."],
            define_macros=[("Py_LIMITED_API", "0x03090000")],
            py_limited_api=True,
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            libraries=libraries,
        )
    ],
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
