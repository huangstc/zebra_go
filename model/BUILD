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
