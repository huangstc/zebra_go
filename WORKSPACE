# Bazel workspace.

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

#
# Google libs
#
http_archive(
    name = "com_github_gflags_gflags",
    url = "https://github.com/gflags/gflags/archive/master.zip",
    strip_prefix = "gflags-master",
)

http_archive(
    name = "com_github_google_absl",
    url = "https://github.com/abseil/abseil-cpp/archive/master.zip",
    strip_prefix = "abseil-cpp-master",
)

http_archive(
    name = "com_github_google_benchmark",
    url = "https://github.com/google/benchmark/archive/master.zip",
    strip_prefix = "benchmark-master",
)

http_archive(
    name = "com_github_google_glog",
    url = "https://github.com/google/glog/archive/master.zip",
    strip_prefix = "glog-master",
)

http_archive(
    name = "com_github_google_googletest",
    url = "https://github.com/google/googletest/archive/master.zip",
    strip_prefix = "googletest-master",
)

#
# My libs
#
http_archive(
    name = "com_github_huangstc_sgf_lib",
    url = "https://github.com/huangstc/sgf_parser/archive/0.12c.zip",
    strip_prefix = "sgf_parser-0.12c",
)
