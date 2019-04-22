tf_absl_libs = [
    "@com_github_google_absl//absl/algorithm",
    "@com_github_google_absl//absl/container:inlined_vector",
    "@com_github_google_absl//absl/types:span",
]

cc_library(
    name = "feature_converter",
    srcs = ["feature_converter.cc"],
    hdrs = ["feature_converter.h"],
    deps = [
      ":tensorflow_dynamic",
      "//engine:go_game",
      "@com_github_google_absl//absl/memory",
      "@com_github_google_absl//absl/strings",
      "@com_github_google_glog//:glog",
  ] + tf_absl_libs
)

cc_test(
    name = "feature_converter_test",
    srcs = ["feature_converter_test.cc"],
    deps = [
      ":feature_converter",
      ":tensorflow_dynamic",
      "//engine:go_game",
      "//engine:sgf_utils",
      "@com_github_google_absl//absl/memory",
      "@com_github_google_glog//:glog",
      "@com_github_google_googletest//:gtest_main",
    ],
)

cc_binary(
    name = "gen_dataset",
    srcs = ["gen_dataset.cc"],
    deps = [
      ":feature_converter",
      ":tensorflow_dynamic",
      "//engine:sgf_utils",
      "@com_github_gflags_gflags//:gflags",
      "@com_github_google_absl//absl/strings",
      "@com_github_google_absl//absl/strings:str_format",
      "@com_github_google_glog//:glog",
    ],
)

cc_library(
    name = "tf_client",
    srcs = ["tf_client.cc"],
    hdrs = ["tf_client.h"],
    deps = [
      ":feature_converter",
      ":tensorflow_dynamic",
      "//engine:go_game",
      "//engine:utils",
      "@com_github_google_absl//absl/memory",
      "@com_github_google_absl//absl/strings",
      "@com_github_google_absl//absl/synchronization",
      "@com_github_google_absl//absl/time",
      "@com_github_google_glog//:glog",
      "@com_github_gflags_gflags//:gflags",
    ] + tf_absl_libs,
    visibility=["//visibility:public"],
)

cc_test(
  name = "tf_client_benchmark",
  srcs = ["tf_client_benchmark.cc"],
  deps = [
    ":tf_client",
    "//engine:go_game",
    "//engine:sgf_utils",
    "@com_github_google_googletest//:gtest_main",
    "@com_github_google_benchmark//:benchmark_main",
    "@com_github_huangstc_sgf_lib//:sgf_parser",
  ],
  data = [
    "//testdata:sgf_files",
    "//testdata:models",
  ],
)

cc_binary(
    name = "eval",
    srcs = ["eval.cc"],
    deps = [
      ":tensorflow_dynamic",
      ":tf_client",
      "//engine:sgf_utils",
      "//engine:utils",
      "@com_github_gflags_gflags//:gflags",
      "@com_github_google_absl//absl/memory",
      "@com_github_google_absl//absl/strings",
      "@com_github_google_glog//:glog",
    ] + tf_absl_libs,
)

cc_library(
    name = "tensorflow_dynamic",
    srcs = [
        "tensorflow/libtensorflow_framework.so",
        "tensorflow/libtensorflow_cc.so",
    ],
    hdrs = glob(
        ["tensorflow/**/*"],
        exclude = [
            "**/*.a",
            "**/*.so",
            "tensorflow/flatbuffers/**",
            "tensorflow/tensorflow/contrib/lite/**",
        ],
    ),
    includes = [
        "tensorflow",
    ],
    tags = ["manual"],
)
