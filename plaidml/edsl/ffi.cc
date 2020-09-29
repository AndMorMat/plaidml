// Copyright 2019 Intel Corporation.

#include "plaidml/edsl/ffi.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

#include "plaidml/core/internal.h"
#include "plaidml/edsl/derivs.h"
#include "pmlc/util/logging.h"

#include "pmlc/ast/ast.h"
#include "pmlc/ast/builder.h"
#include "pmlc/compiler/registry.h"
#include "pmlc/util/enums.h"
#include "pmlc/util/env.h"

using plaidml::core::convertFromDataType;
using plaidml::core::convertIntoDataType;
using plaidml::core::ffi_wrap;
using plaidml::core::ffi_wrap_void;
using pmlc::compiler::Program;
using pmlc::util::AggregationKind;
using pmlc::util::CombinationKind;

namespace ast = pmlc::ast;

namespace {

AggregationKind getAggregationKind(plaidml_agg_op agg_op) {
  switch (agg_op) {
    case PLAIDML_AGG_OP_ASSIGN:
      return AggregationKind::assign;
    case PLAIDML_AGG_OP_MAX:
      return AggregationKind::max;
    case PLAIDML_AGG_OP_MIN:
      return AggregationKind::min;
    case PLAIDML_AGG_OP_PROD:
      return AggregationKind::mul;
    case PLAIDML_AGG_OP_SUM:
      return AggregationKind::add;
    default:
      break;
  }
  throw std::runtime_error("Unsupported agg_op");
}

CombinationKind getCombinationKind(plaidml_combo_op combo_op) {
  switch (combo_op) {
    case PLAIDML_COMBO_OP_ADD:
      return CombinationKind::add;
    case PLAIDML_COMBO_OP_COND:
      return CombinationKind::cond;
    case PLAIDML_COMBO_OP_EQ:
      return CombinationKind::eq;
    case PLAIDML_COMBO_OP_MUL:
      return CombinationKind::mul;
    case PLAIDML_COMBO_OP_NONE:
      return CombinationKind::none;
    default:
      break;
  }
  throw std::runtime_error("Unsupported combo_op");
}

ast::AffineOp getAffineOp(plaidml_int_op op) {
  switch (op) {
    case PLAIDML_INT_OP_ADD:
      return ast::AffineOp::Add;
    case PLAIDML_INT_OP_DIV:
      return ast::AffineOp::Div;
    case PLAIDML_INT_OP_MUL:
      return ast::AffineOp::Mul;
    case PLAIDML_INT_OP_NEG:
      return ast::AffineOp::Neg;
    case PLAIDML_INT_OP_SUB:
      return ast::AffineOp::Sub;
    case PLAIDML_INT_OP_MAX:
      return ast::AffineOp::Max;
    case PLAIDML_INT_OP_MIN:
      return ast::AffineOp::Min;
  }
  throw std::runtime_error("Unknown polynomial op");
}

}  // namespace

extern "C" {

struct plaidml_expr {
  ast::ExprNodePtr node;
};

struct plaidml_poly_expr {
  ast::PolyNodePtr node;
};

struct plaidml_dim_expr {
  ast::DimNodePtr node;
};

void plaidml_edsl_init(  //
    plaidml_error* err) {
  static std::once_flag is_initialized;
  ffi_wrap_void(err, [&] {
    std::call_once(is_initialized, []() {
      IVLOG(1, "plaidml_edsl_init");
      plaidml::edsl::RegisterDerivs();
    });
  });
}

void plaidml_expr_free(  //
    plaidml_error* err,  //
    plaidml_expr* expr) {
  ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_expr_free> " << expr->node->str());
    delete expr;
  });
}

void* plaidml_expr_ptr(  //
    plaidml_error* err,  //
    plaidml_expr* expr) {
  return ffi_wrap<void*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_ptr");
    return expr->node.get();
  });
}

plaidml_datatype plaidml_expr_get_dtype(  //
    plaidml_error* err,                   //
    plaidml_expr* expr) {
  return ffi_wrap<plaidml_datatype>(err, PLAIDML_DATA_INVALID, [&] {
    IVLOG(3, "plaidml_expr_get_dtype");
    return convertIntoDataType(expr->node->getShape().elementType);
  });
}

size_t plaidml_expr_get_rank(  //
    plaidml_error* err,        //
    plaidml_expr* expr) {
  return ffi_wrap<size_t>(err, 0, [&] { return expr->node->getShape().getRank(); });
}

plaidml_shape* plaidml_expr_get_shape(  //
    plaidml_error* err,                 //
    plaidml_expr* expr) {
  return ffi_wrap<plaidml_shape*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_get_shape");
    if (!expr) {
      throw std::runtime_error(
          "Cannot compute shape of null expr. Perhaps you requested the shape of an unassigned tensor?");
    }
    return new plaidml_shape{expr->node->getShape()};
  });
}

void plaidml_expr_bind_shape(  //
    plaidml_error* err,        //
    plaidml_expr* expr,        //
    plaidml_shape* shape) {
  return ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_expr_bind_shape");
    auto* node = llvm::dyn_cast<ast::ExprNodeInput>(expr->node.get());
    if (!node) {
      throw std::bad_cast();
    }
    node->shape = shape->shape;
  });
}

void plaidml_expr_bind_dims(  //
    plaidml_error* err,       //
    plaidml_expr* expr,       //
    size_t rank,              //
    plaidml_dim_expr** dims) {
  return ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_expr_bind_dims> " << expr->node->str());
    for (size_t i = 0; i < rank; i++) {
      dims[i]->node = std::make_shared<ast::DimNodeRef>(expr->node, i);
    }
  });
}

plaidml_string* plaidml_expr_repr(  //
    plaidml_error* err,             //
    plaidml_expr* expr) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {
    // IVLOG(3, "plaidml_expr_repr");
    return new plaidml_string{expr->node->str()};
  });
}

plaidml_expr* plaidml_expr_dim(  //
    plaidml_error* err,          //
    plaidml_dim_expr* expr) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_dim");
    return new plaidml_expr{std::make_shared<ast::ExprNodeDim>(expr->node)};
  });
}

plaidml_expr* plaidml_expr_input(  //
    plaidml_error* err,            //
    plaidml_shape* shape,          //
    const char* name) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_input");
    return new plaidml_expr{std::make_shared<ast::ExprNodeInput>(shape->shape, name)};
  });
}

plaidml_expr* plaidml_expr_constant(  //
    plaidml_error* err,               //
    plaidml_shape* shape,             //
    plaidml_buffer* buffer,           //
    const char* name) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_constant");
    return new plaidml_expr{std::make_shared<ast::ExprNodeConstTensor>(shape->shape, buffer->buffer, name)};
  });
}

plaidml_expr* plaidml_expr_clone(  //
    plaidml_error* err,            //
    plaidml_expr* expr) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_clone> " << expr->node->str());
    return new plaidml_expr{expr->node};
  });
}

plaidml_dim_expr* plaidml_expr_get_dim(  //
    plaidml_error* err,                  //
    plaidml_expr* expr) {
  return ffi_wrap<plaidml_dim_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_get_dim> " << expr->node->str());
    auto* node = llvm::dyn_cast<ast::ExprNodeDim>(expr->node.get());
    if (!node) {
      throw std::bad_cast();
    }
    return new plaidml_dim_expr{node->dim};
  });
}

plaidml_expr* plaidml_expr_uint(  //
    plaidml_error* err,           //
    uint64_t value) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_uint> " << value);
    return new plaidml_expr{std::make_shared<ast::ExprNodeConstUnsigned>(value)};
  });
}

plaidml_expr* plaidml_expr_int(  //
    plaidml_error* err,          //
    int64_t value) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_int> " << value);
    return new plaidml_expr{std::make_shared<ast::ExprNodeConstSigned>(value)};
  });
}

plaidml_expr* plaidml_expr_float(  //
    plaidml_error* err,            //
    double value) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_float");
    return new plaidml_expr{std::make_shared<ast::ExprNodeConstFloat>(value)};
  });
}

plaidml_expr* plaidml_expr_cast(  //
    plaidml_error* err,           //
    plaidml_expr* expr,           //
    plaidml_datatype dtype) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_cast");
    return new plaidml_expr{std::make_shared<ast::ExprNodeCast>(convertFromDataType(dtype), expr->node)};
  });
}

plaidml_expr* plaidml_expr_element(  //
    plaidml_error* err,              //
    plaidml_expr* expr,              //
    size_t ordinal) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_element");
    return new plaidml_expr{std::make_shared<ast::ExprNodeElement>(expr->node, ordinal)};
  });
}

plaidml_expr* plaidml_expr_trace(  //
    plaidml_error* err,            //
    plaidml_expr* expr,            //
    const char* msg) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_trace");
    return new plaidml_expr{std::make_shared<ast::ExprNodeTrace>(expr->node, msg)};
  });
}

plaidml_expr* plaidml_expr_intrinsic(  //
    plaidml_error* err,                //
    const char* fn,                    //
    size_t nargs,                      //
    plaidml_expr** args) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_intrinsic: " << fn);
    std::vector<ast::ExprNodePtr> operands(nargs);
    for (size_t i = 0; i < nargs; i++) {
      operands[i] = args[i]->node;
    }
    return new plaidml_expr{std::make_shared<ast::ExprNodeIntrinsic>(fn, operands)};
  });
}

plaidml_expr* plaidml_expr_contraction(  //
    plaidml_error* err,                  //
    plaidml_agg_op agg_op,               //
    plaidml_combo_op combo_op,           //
    size_t rank,                         //
    plaidml_poly_expr** idxs,            //
    plaidml_dim_expr** dims,             //
    plaidml_expr* init,                  //
    bool simplify,                       //
    const char* name) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_expr_contraction");
    auto node = std::make_shared<ast::ExprNodeContraction>(name);
    node->aggKind = getAggregationKind(agg_op);
    node->comboKind = getCombinationKind(combo_op);
    for (size_t i = 0; i < rank; i++) {
      node->sinkDims.push_back(dims[i]->node);
      node->sinkIdxs.push_back(idxs[i]->node);
    }
    node->simplify = simplify;
    if (init) {
      node->init = init->node;
    }
    return new plaidml_expr{node};
  });
}

void plaidml_contraction_add_operand(  //
    plaidml_error* err,                //
    plaidml_expr* expr,                //
    plaidml_expr* ref,                 //
    size_t rank,                       //
    plaidml_poly_expr** idxs) {
  ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_contraction_add_operand");
    auto* node = llvm::dyn_cast<ast::ExprNodeContraction>(expr->node.get());
    if (!node) {
      throw std::bad_cast();
    }
    ast::PolyMap map;
    map.ref = ref->node;
    for (size_t i = 0; i < rank; i++) {
      map.idxs.push_back(idxs[i]->node);
    }
    node->srcs.emplace_back(map);
  });
}

void plaidml_contraction_add_constraint(  //
    plaidml_error* err,                   //
    plaidml_expr* expr,                   //
    plaidml_poly_expr* lhs,               //
    plaidml_dim_expr* rhs) {
  ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_contraction_add_constraint");
    auto* node = llvm::dyn_cast<ast::ExprNodeContraction>(expr->node.get());
    if (!node) {
      throw std::bad_cast();
    }
    node->constraints.emplace_back(lhs->node, rhs->node);
  });
}

void plaidml_poly_expr_free(plaidml_error* err, plaidml_poly_expr* expr) {
  ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_poly_expr_free> " << expr->node->str());
    delete expr;
  });
}

plaidml_string* plaidml_poly_expr_repr(  //
    plaidml_error* err,                  //
    plaidml_poly_expr* expr) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_poly_expr_repr");
    return new plaidml_string{expr->node->str()};
  });
}

plaidml_poly_expr* plaidml_poly_expr_dim(  //
    plaidml_error* err,                    //
    plaidml_dim_expr* expr) {
  return ffi_wrap<plaidml_poly_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_poly_expr_dim");
    return new plaidml_poly_expr{std::make_shared<ast::PolyNodeDim>(expr->node)};
  });
}

plaidml_poly_expr* plaidml_poly_expr_index(  //
    plaidml_error* err,                      //
    const char* name) {
  return ffi_wrap<plaidml_poly_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_poly_expr_index");
    return new plaidml_poly_expr{std::make_shared<ast::PolyNodeIndex>(name)};
  });
}

plaidml_poly_expr* plaidml_poly_expr_literal(  //
    plaidml_error* err,                        //
    int64_t value) {
  return ffi_wrap<plaidml_poly_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_poly_expr_literal> " << value);
    return new plaidml_poly_expr{std::make_shared<ast::PolyNodeLiteral>(value)};
  });
}

plaidml_poly_expr* plaidml_poly_expr_op(  //
    plaidml_error* err,                   //
    plaidml_int_op op,                    //
    size_t nargs,                         //
    plaidml_poly_expr** args) {
  return ffi_wrap<plaidml_poly_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_poly_expr_op> " << op);
    std::vector<ast::PolyNodePtr> operands(nargs);
    for (size_t i = 0; i < nargs; i++) {
      operands[i] = args[i]->node;
    }
    return new plaidml_poly_expr{std::make_shared<ast::PolyNodeOp>(getAffineOp(op), operands)};
  });
}

void plaidml_dim_expr_free(  //
    plaidml_error* err,      //
    plaidml_dim_expr* expr) {
  ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_dim_expr_free> " << expr->node->str());
    delete expr;
  });
}

plaidml_string* plaidml_dim_expr_repr(  //
    plaidml_error* err,                 //
    plaidml_dim_expr* expr) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_dim_expr_repr");
    return new plaidml_string{expr->node->str()};
  });
}

plaidml_dim_expr* plaidml_dim_expr_none(  //
    plaidml_error* err                    //
) {
  return ffi_wrap<plaidml_dim_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_dim_expr_none");
    return new plaidml_dim_expr{std::make_shared<ast::DimNodeNone>()};
  });
}

plaidml_dim_expr* plaidml_dim_expr_int(  //
    plaidml_error* err,                  //
    int64_t value) {
  return ffi_wrap<plaidml_dim_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_dim_expr_int> " << value);
    return new plaidml_dim_expr{std::make_shared<ast::DimNodeLiteral>(value)};
  });
}

// int64_t plaidml_dim_expr_get_int(  //
//     plaidml_error* err,            //
//     plaidml_dim_expr* expr) {
//   return ffi_wrap<int64_t>(err, 0, [&] {
//     IVLOG(3, "plaidml_dim_expr_get_int");
//     if (!expr) {
//       throw std::runtime_error("plaidml_dim_expr_get_int can only be used on an integer value");
//     }
//     // TODO
//     // return expr->node->builder->GetIntegerValue(expr->node->value);
//     return 0;
//   });
// }

plaidml_dim_expr* plaidml_dim_expr_op(  //
    plaidml_error* err,                 //
    plaidml_int_op op,                  //
    size_t nargs,                       //
    plaidml_dim_expr** args) {
  return ffi_wrap<plaidml_dim_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_dim_expr_op> " << op);
    std::vector<ast::DimNodePtr> operands(nargs);
    for (size_t i = 0; i < nargs; i++) {
      operands[i] = args[i]->node;
    }
    return new plaidml_dim_expr{std::make_shared<ast::DimNodeOp>(getAffineOp(op), operands)};
  });
}

void plaidml_tuple_free(  //
    plaidml_error* err,   //
    plaidml_tuple* tuple) {
  ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_tuple_free");
    for (size_t i = 0; i < tuple->size; i++) {
      delete tuple->elts[i];
    }
    delete[] tuple->elts;
    delete tuple;
  });
}

void plaidml_value_free(  //
    plaidml_error* err,   //
    plaidml_value* value) {
  ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_value_free: " << value->node->str());
    delete value;
  });
}

plaidml_value* plaidml_value_clone(  //
    plaidml_error* err,              //
    plaidml_value* value) {
  return ffi_wrap<plaidml_value*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_value_clone");
    return new plaidml_value{*value};
  });
}

plaidml_value_kind plaidml_value_get_kind(  //
    plaidml_error* err,                     //
    plaidml_value* value) {
  return ffi_wrap<plaidml_value_kind>(err, PLAIDML_VALUE_NONE, [&] {
    IVLOG(3, "plaidml_value_get_kind");
    return llvm::TypeSwitch<ast::VarNode*, plaidml_value_kind>(value->node.get())
        .Case<ast::VarNodeDim>([](auto*) { return PLAIDML_VALUE_DIM; })
        .Case<ast::VarNodeExpr>([](auto*) { return PLAIDML_VALUE_EXPR; })
        .Case<ast::VarNodeFloat>([](auto*) { return PLAIDML_VALUE_FLOAT; })
        .Case<ast::VarNodeInt>([](auto*) { return PLAIDML_VALUE_INT; })
        .Case<ast::VarNodeNone>([](auto*) { return PLAIDML_VALUE_NONE; })
        .Case<ast::VarNodeString>([](auto*) { return PLAIDML_VALUE_STR; })
        .Case<ast::VarNodeTuple>([](auto*) { return PLAIDML_VALUE_TUPLE; })
        .Default([](auto*) -> plaidml_value_kind { throw std::bad_cast(); });
  });
}

plaidml_value* plaidml_value_none(  //
    plaidml_error* err              //
) {
  return ffi_wrap<plaidml_value*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_value_none");
    return new plaidml_value{std::make_shared<ast::VarNodeNone>()};
  });
}

plaidml_value* plaidml_value_int(  //
    plaidml_error* err,            //
    int64_t value) {
  return ffi_wrap<plaidml_value*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_value_int> " << value);
    return new plaidml_value{std::make_shared<ast::VarNodeInt>(value)};
  });
}

plaidml_value* plaidml_value_dim(  //
    plaidml_error* err,            //
    plaidml_dim_expr* expr) {
  return ffi_wrap<plaidml_value*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_value_dim");
    if (!expr) {
      throw std::runtime_error("plaidml_value_dim requires non-null expr");
    }
    return new plaidml_value{std::make_shared<ast::VarNodeDim>(expr->node)};
  });
}

plaidml_value* plaidml_value_expr(  //
    plaidml_error* err,             //
    plaidml_expr* expr) {
  return ffi_wrap<plaidml_value*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_value_expr");
    if (!expr) {
      throw std::runtime_error("plaidml_value_expr requires non-null expr");
    }
    return new plaidml_value{std::make_shared<ast::VarNodeExpr>(expr->node)};
  });
}

plaidml_value* plaidml_value_float(  //
    plaidml_error* err,              //
    double value) {
  return ffi_wrap<plaidml_value*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_value_float");
    return new plaidml_value{std::make_shared<ast::VarNodeFloat>(value)};
  });
}

plaidml_value* plaidml_value_str(  //
    plaidml_error* err,            //
    const char* value) {
  return ffi_wrap<plaidml_value*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_value_str> " << value);
    return new plaidml_value{std::make_shared<ast::VarNodeString>(value)};
  });
}

plaidml_value* plaidml_value_tuple(  //
    plaidml_error* err,              //
    size_t size,                     //
    plaidml_value** elts) {
  return ffi_wrap<plaidml_value*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_value_tuple: " << size);
    auto tuple = std::make_shared<ast::VarNodeTuple>();
    for (size_t i = 0; i < size; i++) {
      tuple->values.push_back(elts[i]->node);
    }
    return new plaidml_value{tuple};
  });
}

plaidml_dim_expr* plaidml_value_dim_get(  //
    plaidml_error* err,                   //
    plaidml_value* value) {
  return ffi_wrap<plaidml_dim_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_value_dim_get");
    auto node = std::dynamic_pointer_cast<ast::VarNodeDim>(value->node);
    if (!node) {
      throw std::bad_cast();
    }
    return new plaidml_dim_expr{node->value};
  });
}

plaidml_expr* plaidml_value_expr_get(  //
    plaidml_error* err,                //
    plaidml_value* value) {
  return ffi_wrap<plaidml_expr*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_value_expr_get");
    auto node = std::dynamic_pointer_cast<ast::VarNodeExpr>(value->node);
    if (!node) {
      throw std::bad_cast();
    }
    return new plaidml_expr{node->value};
  });
}

double plaidml_value_float_get(  //
    plaidml_error* err,          //
    plaidml_value* value) {
  return ffi_wrap<double>(err, 0, [&] {
    IVLOG(3, "plaidml_value_float_get");
    auto node = std::dynamic_pointer_cast<ast::VarNodeFloat>(value->node);
    if (!node) {
      throw std::bad_cast();
    }
    return node->value;
  });
}

int64_t plaidml_value_int_get(  //
    plaidml_error* err,         //
    plaidml_value* value) {
  return ffi_wrap<int64_t>(err, 0, [&] {
    IVLOG(3, "plaidml_value_int_get");
    auto node = std::dynamic_pointer_cast<ast::VarNodeInt>(value->node);
    if (!node) {
      throw std::bad_cast();
    }
    return node->value;
  });
}

plaidml_tuple* plaidml_value_tuple_get(  //
    plaidml_error* err,                  //
    plaidml_value* value) {
  return ffi_wrap<plaidml_tuple*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_value_tuple_get");
    auto node = std::dynamic_pointer_cast<ast::VarNodeTuple>(value->node);
    if (!node) {
      throw std::bad_cast();
    }
    auto size = node->values.size();
    auto elts = new plaidml_value*[size];
    for (size_t i = 0; i < size; i++) {
      elts[i] = new plaidml_value{node->values[i]};
    }
    return new plaidml_tuple{size, elts};
  });
}

plaidml_string* plaidml_value_str_get(  //
    plaidml_error* err,                 //
    plaidml_value* value) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_value_str_get");
    auto node = std::dynamic_pointer_cast<ast::VarNodeString>(value->node);
    if (!node) {
      throw std::bad_cast();
    }
    return new plaidml_string{node->value};
  });
}

plaidml_string* plaidml_value_repr(  //
    plaidml_error* err,              //
    plaidml_value* value) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_value_repr");
    return new plaidml_string{value->node->str()};
  });
}

// plaidml_program* plaidml_compile(  //
//     plaidml_error* err,            //
//     const char* name,              //
//     const char* target,            //
//     size_t noutputs,               //
//     plaidml_expr** raw_outputs,    //
//     size_t nupdates,               //
//     plaidml_expr** src_updates,    //
//     plaidml_expr** dst_updates,    //
//     plaidml_datatype floatx,       //
//     plaidml_datatype intx,         //
//     bool debug,                    //
//     plaidml_program_args** raw_args) {
//   return ffi_wrap<plaidml_program*>(err, nullptr, [&] {
//     IVLOG(3, "plaidml_compile");
//     IVLOG(5, "  plaidml_compile>> noutputs: " << noutputs << ", nupdates: " << nupdates);
//     ProgramMutations mutations;
//     for (size_t i = 0; i < noutputs; i++) {
//       if (!raw_outputs[i]) {
//         throw std::runtime_error("Undefined output in plaidml_compile");
//       }
//       mutations.outputs.emplace_back(raw_outputs[i]->value);
//     }
//     for (size_t i = 0; i < nupdates; i++) {
//       if (!src_updates[i]) {
//         throw std::runtime_error("Undefined update src in plaidml_compile");
//       }
//       if (!dst_updates[i]) {
//         throw std::runtime_error("Undefined update dst in plaidml_compile");
//       }
//       mutations.updates.emplace(ProgramUpdate{src_updates[i]->value, dst_updates[i]->value});
//     }

//     auto ctx = GlobalContext::get();
//     auto floatType = convertFromDataType(floatx, ctx->getContext());
//     if (!floatType.isa<mlir::FloatType>()) {
//       throw std::runtime_error("Invalid floatx in plaidml_compile");
//     }

//     auto intType = convertFromDataType(intx, ctx->getContext());
//     if (!intType.isa<mlir::IntegerType>()) {
//       throw std::runtime_error("Invalid intx in plaidml_compile");
//     }

//     auto program = ctx->MakeProgram(name, mutations, floatType, intType);
//     auto ret = new plaidml_program{program};
//     auto nargs = ret->program->arguments.size();
//     auto args = new plaidml_program_arg[nargs];
//     for (unsigned i = 0; i < nargs; i++) {
//       args[i].is_input = ret->program->arguments[i].isInput;
//       args[i].tensor = new plaidml_expr{ret->program->arguments[i].value};
//       args[i].shape = new plaidml_shape{ret->program->arguments[i].shape};
//       if (ret->program->arguments[i].buffer) {
//         args[i].buffer = new plaidml_buffer{ret->program->arguments[i].buffer};
//       } else {
//         args[i].buffer = nullptr;
//       }
//     }
//     *raw_args = new plaidml_program_args{nargs, args};
//     auto dumpDir = pmlc::util::getEnvVar("PLAIDML_DUMP");
//     program->compile(target, /*collectPasses=*/debug, /*dumpDir=*/dumpDir);
//     return ret;
//   });
// }

plaidml_strings* plaidml_targets_get(  //
    plaidml_error* err) {
  return ffi_wrap<plaidml_strings*>(err, nullptr, [&] {
    const auto& targets = pmlc::compiler::listTargets();
    auto strs = new plaidml_string*[targets.size()];
    for (unsigned i = 0; i < targets.size(); i++) {
      strs[i] = new plaidml_string{targets[i].str()};
    }
    return new plaidml_strings{targets.size(), strs};
  });
}

plaidml_program* plaidml_build(  //
    plaidml_error* err,          //
    const char* name,            //
    size_t ninputs,              //
    plaidml_expr** inputs,       //
    size_t noutputs,             //
    plaidml_expr** outputs) {
  return ffi_wrap<plaidml_program*>(err, nullptr, [&] {
    ast::ProgramArguments args;
    for (size_t i = 0; i < ninputs; i++) {
      args.inputs.push_back(inputs[i]->node);
    }
    for (size_t i = 0; i < noutputs; i++) {
      args.outputs.push_back(outputs[i]->node);
    }
    return new plaidml_program{ast::buildProgram(name, args)};
  });
}

}  // extern "C"
