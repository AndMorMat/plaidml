# Copyright 2019 Intel Corporation

import logging

import numpy as np

import plaidml
import plaidml.edsl as edsl
import plaidml.settings
from plaidml.ffi import ForeignObject, decode_str, ffi, ffi_call, lib

logger = logging.getLogger(__name__)


def __init():
    """
    Initializes the PlaidML Execution API.
    """
    ffi_call(lib.plaidml_exec_init)


ffi.init_once(__init, 'plaidml_exec_init')


def list_devices():
    return plaidml.get_strings(lib.plaidml_devices_get)


class Executable(ForeignObject):
    __ffi_del__ = lib.plaidml_executable_free

    def __init__(self, program, inputs=[], outputs=[], device=''):
        inputs = [x.as_ptr() for x in inputs]
        outputs = [x.as_ptr() for x in outputs]
        ffi_obj = ffi_call(
            lib.plaidml_jit,
            program.as_ptr(),
            device.encode(),
            len(inputs),
            inputs,
            len(outputs),
            outputs,
        )
        super(Executable, self).__init__(ffi_obj)

    def run(self):
        self._methodcall(lib.plaidml_executable_run)


class Runner(object):

    def __init__(self, program, device=''):
        program.compile()
        self.program = program
        self.inputs = [plaidml.Buffer(shape) for shape in program.inputs]
        self.outputs = [plaidml.Buffer(shape) for shape in program.outputs]
        self.executable = Executable(program, self.inputs, self.outputs, device=device)

    def run(self, inputs):
        for ndarray, buffer in zip(inputs, self.inputs):
            buffer.copy_from_ndarray(ndarray)
        self.executable.run()
        return [buffer.as_ndarray() for buffer in self.outputs]


def run(program, inputs, device=''):
    return Runner(program, device=device).run(inputs)
