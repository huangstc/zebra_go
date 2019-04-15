
cc_library(
    name = "go_game",
    srcs = ["go_game.cc"],
    hdrs = ["go_game.h"],
    deps = [
      "@com_github_google_absl//absl/memory",
      "@com_github_google_absl//absl/strings",
      "@com_github_google_glog//:glog",
    ],
    visibility=["//visibility:public"],
)

cc_test(
    name = "go_game_test",
    srcs = ["go_game_test.cc"],
    deps = [
      ":go_game",
      ":sgf_utils",
      "@com_github_google_googletest//:gtest_main",
    ],
    data = [
      "//testdata:sgf_files",
    ],
)

cc_test(
    name = "go_game_benchmark",
    srcs = ["go_game_benchmark.cc"],
    deps = [
      ":go_game",
      ":sgf_utils",
      "@com_github_google_googletest//:gtest_main",
      "@com_github_google_benchmark//:benchmark_main",
      "@com_github_huangstc_sgf_lib//:sgf_parser",
    ],
    data = [
      "//testdata:sgf_files",
    ],
)

cc_library(
    name = "sgf_utils",
    srcs = ["sgf_utils.cc"],
    hdrs = ["sgf_utils.h"],
    deps = [
      ":go_game",
      "@com_github_google_absl//absl/strings",
      "@com_github_google_glog//:glog",
      "@com_github_huangstc_sgf_lib//:sgf_parser",
    ],
    visibility=["//visibility:public"],
)
