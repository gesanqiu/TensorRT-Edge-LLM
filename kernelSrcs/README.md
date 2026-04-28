# CuTe DSL Artifact Generation

`kernelSrcs/build_cutedsl.py` is the unified entry point for generating local
CuTe DSL artifacts consumed by `cmake/CuteDsl.cmake`.

## What It Generates

Running the script produces a local artifact directory under:

```text
cpp/kernels/cuteDSLArtifact/<arch>/<artifact_tag>/
  libcutedsl_<arch>.a
  metadata.json
  include/
    cutedsl_all.h
    ...
```

`artifact_tag` is currently `sm_<NN>`, for example `sm_80`, `sm_110`, or
`sm_121`.

These artifacts are local build inputs for CMake. They are not intended to be
checked into git by default.

## Typical Workflow

1. Install the CuTe DSL Python dependencies on a supported machine.
2. Run `python kernelSrcs/build_cutedsl.py ...` for the desired kernel group and
   GPU SM.
3. Build the C++ project with `-DENABLE_CUTE_DSL=...`.
4. If multiple artifact tags exist for the same CPU architecture, pass
   `-DCUTE_DSL_ARTIFACT_TAG=<tag>` to CMake.

## Common Commands

From the repository root:

```bash
# Build all groups supported by the current GPU
python kernelSrcs/build_cutedsl.py

# Build one group for a specific target SM
python kernelSrcs/build_cutedsl.py --kernels gdn --gpu_arch sm_87
python kernelSrcs/build_cutedsl.py --kernels fmha --gpu_arch sm_110 --arch aarch64
python kernelSrcs/build_cutedsl.py --kernels gemm --gpu_arch sm_121 --arch aarch64
```

## CMake Integration

Enable one or more groups with `ENABLE_CUTE_DSL`:

```bash
cmake .. -DENABLE_CUTE_DSL=fmha
cmake .. -DENABLE_CUTE_DSL=gdn
cmake .. -DENABLE_CUTE_DSL=gemm
cmake .. -DENABLE_CUTE_DSL="fmha;gdn;gemm"
```

When the same CPU architecture has multiple local artifact tags, select one
explicitly:

```bash
cmake .. -DENABLE_CUTE_DSL=gemm -DCUTE_DSL_ARTIFACT_TAG=sm_110
cmake .. -DENABLE_CUTE_DSL=gemm -DCUTE_DSL_ARTIFACT_TAG=sm_121
```

`EMBEDDED_TARGET=gb10`, `auto-thor`, `jetson-thor`, and `jetson-orin` map to a
default artifact tag when unambiguous. `thor-all` requires an explicit
`CUTE_DSL_ARTIFACT_TAG`.

## Group-Specific Details

See the group-specific READMEs for kernel coverage and standalone testing:

- `kernelSrcs/gdn_cutedsl/README.md`
- `kernelSrcs/fmha_cutedsl_blackwell/README.md`
- `kernelSrcs/gemm_cutedsl/README.md`
