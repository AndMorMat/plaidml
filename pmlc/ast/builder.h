// Copyright 2020 Intel Corporation

#pragma once

#include <memory>
#include <vector>

#include "llvm/ADT/StringRef.h"

#include "pmlc/ast/ast.h"
#include "pmlc/compiler/program.h"

namespace pmlc::ast {

struct ProgramUpdate {
  ExprNodePtr src;
  ExprNodePtr dst;
};

struct ProgramArguments {
  std::vector<ExprNodePtr> inputs;
  std::vector<ExprNodePtr> outputs;
  std::vector<ProgramUpdate> updates;
};

std::shared_ptr<compiler::Program> buildProgram(llvm::StringRef name,
                                                const ProgramArguments &args);

} // namespace pmlc::ast
