// Copyright 2019 Intel Corporation.

#include "plaidml/exec/ffi.h"

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "plaidml/core/internal.h"
#include "plaidml/core/settings.h"
#include "pmlc/rt/device_id.h"
#include "pmlc/rt/executable.h"
#include "pmlc/rt/runtime_registry.h"
#include "pmlc/util/env.h"
#include "pmlc/util/logging.h"

using plaidml::core::ffi_vector;
using plaidml::core::ffi_wrap;
using plaidml::core::ffi_wrap_void;
using pmlc::rt::Device;
using pmlc::rt::EngineKind;
using pmlc::rt::Executable;
using pmlc::rt::getDeviceIDs;
using namespace mlir;  // NOLINT[build/namespaces]

extern "C" {

struct plaidml_executable {
  std::unique_ptr<Executable> exec;
};

void plaidml_exec_init(  //
    plaidml_error* err) {
  ffi_wrap_void(err, [&] {  //
    IVLOG(1, "plaidml_exec_init");
    pmlc::rt::initRuntimes();
  });
}

plaidml_strings* plaidml_devices_get(  //
    plaidml_error* err) {
  return ffi_wrap<plaidml_strings*>(err, nullptr, [&] {  //
    return ffi_vector<plaidml_strings, plaidml_string>(getDeviceIDs());
  });
}

plaidml_executable* plaidml_jit(  //
    plaidml_error* err,           //
    plaidml_program* program,     //
    const char* raw_device,       //
    size_t ninputs,               //
    plaidml_buffer** inputs,      //
    size_t noutputs,              //
    plaidml_buffer** outputs) {
  return ffi_wrap<plaidml_executable*>(err, nullptr, [&] {
    std::string device(raw_device);
    if (device.empty()) {
      device = plaidml::core::Settings::Instance()->get("PLAIDML_DEVICE");
    }
    IVLOG(1, "JITing for device: " << device);
    auto exec = std::make_unique<plaidml_executable>();
    std::vector<void*> bufptrs(ninputs + noutputs);
    for (unsigned i = 0; i < ninputs; i++) {
      bufptrs[i] = inputs[i]->buffer->data();
    }
    for (unsigned i = 0; i < noutputs; i++) {
      unsigned j = ninputs + i;
      bufptrs[j] = outputs[i]->buffer->data();
    }
    EngineKind kind = EngineKind::OrcJIT;
    auto jit = pmlc::util::getEnvVar("LLVM_JIT");
    if (jit == "ORC") {
      kind = EngineKind::OrcJIT;
    } else if (jit == "MCJIT") {
      kind = EngineKind::MCJIT;
    }
    exec->exec = Executable::fromProgram(program->program, device, bufptrs, kind);
    return exec.release();
  });
}

void plaidml_executable_free(  //
    plaidml_error* err,        //
    plaidml_executable* exec) {
  ffi_wrap_void(err, [&] {  //
    delete exec;
  });
}

void plaidml_executable_run(  //
    plaidml_error* err,       //
    plaidml_executable* exec) {
  ffi_wrap_void(err, [&] {  //
    exec->exec->invoke();
  });
}

}  // extern "C"
