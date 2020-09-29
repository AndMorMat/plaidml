# Copyright 2020 Intel Corporation

load("//bzl:plaidml.bzl", "plaidml_cc_library")

plaidml_cc_library(
    name = "llvm",
    srcs = [
        "builtin.cc",
        "runtime.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//pmlc/compiler",
        "//pmlc/rt",
    ],
    alwayslink = 1,
)
