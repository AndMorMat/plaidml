// Copyright 2020 Intel Corporation

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"

#include "pmlc/util/buffer.h"

namespace pmlc::compiler {

struct ProgramArgument {
  bool isInput;
  mlir::Value value;
  mlir::RankedTensorType shape;
  pmlc::util::BufferPtr buffer;
};

struct PassInfo {
  std::string name;
  std::string ir;
};

struct Program {
  std::string entry;
  std::string tileIR;
  mlir::MLIRContext context;
  mlir::OwningModuleRef module;
  std::vector<mlir::Type> inputs;
  std::vector<mlir::Type> outputs;
  std::vector<PassInfo> passes;

  Program();
  explicit Program(mlir::ModuleOp module);
  explicit Program(mlir::StringRef source);
  explicit Program(std::unique_ptr<llvm::MemoryBuffer> buffer);

  void compile(mlir::StringRef target, bool collectPasses = false,
               mlir::StringRef dumpDir = "");
};

} // namespace pmlc::compiler
