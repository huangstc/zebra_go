#ifndef ZEBRA_GO_ENGINE_SGF_UTILS_H_
#define ZEBRA_GO_ENGINE_SGF_UTILS_H_

#include <memory>
#include <string>

#include "engine/go_game.h"

namespace zebra_go {

// Loads a game record to a GoBoard instance.
// Example input:
//  static const char kSgf[] = R"((
//    ;GM[1]FF[4]CA[UTF-8]AP[test]SZ[9]
//    ;AB[ba][ab][cb][db][bd][cd][ed][ae][de]
//    ;AW[ca][da][bb][eb][bc][dc][ad][dd][ce]
//    ;B[cg];W[gc]
//  ))";
// It may return null in case of any error.
std::unique_ptr<GoBoard> SgfToGoBoard(const std::string& sgf);

// Reads the file content to a string.
std::string ReadFileToString(const std::string& filename);

}  // zebra_go

#endif  // ZEBRA_GO_ENGINE_SGF_UTILS_H_
