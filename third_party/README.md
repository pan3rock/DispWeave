# Bundled dependencies

The default CMake build uses only these files. It does not download dependencies
or discover packages from machine-specific paths. Source/header subsets are
pinned by version and per-file SHA-256 manifests. Run
`python scripts/verify_vendor.py` to verify them; Python is not a build dependency.

| Dependency | Version | License and scope |
| --- | --- | --- |
| Eigen | 3.4.1 | Upstream `Eigen/` headers and all COPYING files; MPL-2.0 and per-file notices. `EIGEN_MPL2_ONLY` enabled. |
| toml11 | 4.4.0 | Upstream include tree and MIT LICENSE. |
| CLI11 | 2.4.2 | Upstream include tree and BSD-3-Clause LICENSE. |
| invLM | 86ad6fa5aaec3f44c6ddcac3d69b27a2f13c2e88 | Selected dispersion and kernel sources and Apache-2.0 LICENSE. One documented local patch. |

See each directory's `provenance.json`, original license and source notices.
The DispWeave top-level license does not replace third-party licenses.
The installed executable is accompanied by license/provenance files under
`share/dispweave/third_party`. Source is distributed in this repository.

The invLM numerical sources were imported from the user's local repository.
The import does not represent an independent authorship claim or certification
of every upstream algorithm. M2 adds unchanged energy-kernel and implicit-derivative sources from the same commit.
