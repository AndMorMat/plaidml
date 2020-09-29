// Copyright 2020 Intel Corporation

#include "pmlc/ast/ast.h"

#include <algorithm>
#include <sstream>

#include "llvm/Support/FormatVariadic.h"

#include "pmlc/ast/ast_ops.h"
#include "pmlc/util/logging.h"

namespace pmlc::ast {

using pmlc::util::DataType;

static llvm::StringRef getAffineOpStr(AffineOp op) {
  switch (op) {
  case AffineOp::Add:
    return "add";
  case AffineOp::Div:
    return "div";
  case AffineOp::Max:
    return "max";
  case AffineOp::Min:
    return "min";
  case AffineOp::Mul:
    return "mul";
  case AffineOp::Neg:
    return "neg";
  case AffineOp::Sub:
    return "sub";
  default:
    return "<invalid op>";
  }
  llvm_unreachable("getAffineOpStr");
}

static int64_t getTypeScore(DataType type) {
  return static_cast<int64_t>(type);
}

static DataType promoteTypes(DataType lhs, DataType rhs) {
  return getTypeScore(lhs) > getTypeScore(rhs) ? lhs : rhs;
}

DataType inferElementType(llvm::ArrayRef<TensorShape> shapes) {
  DataType ret = DataType::invalid;
  for (const TensorShape &shape : shapes) {
    ret = promoteTypes(ret, shape.elementType);
  }
  return ret;
}

static DataType inferElementType(util::CombinationKind combo,
                                 llvm::ArrayRef<PolyMap> srcs) {
  IVLOG(1, "inferElementType");
  if (combo == util::CombinationKind::eq) {
    return DataType::i1;
  }
  if (combo == util::CombinationKind::cond) {
    return srcs[2].ref->getShape().elementType;
  }
  llvm::SmallVector<TensorShape, 3> shapes;
  for (const PolyMap &src : srcs) {
    shapes.push_back(src.ref->getShape());
  }
  return inferElementType(shapes);
}

static bool mergeShapes(TensorShape *into, const TensorShape &from,
                        DataType dtype) {
  // To compute the resulting broadcasted shape, we compare operand shapes
  // element-wise: starting with the trailing dimensions, and working our
  // way backward. Two dimensions are compatible when
  //   1. they are equal, or
  //   2. one of them is 1
  // The result shape has the maximum among the two inputs at every
  // dimension index.
  std::vector<int64_t> resultShape;
  const std::vector<int64_t> &shape1 = into->sizes;
  const std::vector<int64_t> &shape2 = from.sizes;
  IVLOG(6, "  Checking compatibility between " << shape1 << " and " << shape2);
  if (shape1.size() > shape2.size()) {
    std::copy(shape1.begin(), shape1.end(), std::back_inserter(resultShape));
  } else {
    std::copy(shape2.begin(), shape2.end(), std::back_inserter(resultShape));
  }

  auto i1 = shape1.rbegin(), e1 = shape1.rend();
  auto i2 = shape2.rbegin(), e2 = shape2.rend();
  auto iR = resultShape.rbegin();

  // Check each dimension is consistent.
  for (; i1 != e1 && i2 != e2; ++i1, ++i2, ++iR) {
    if (*i1 == -1 || *i2 == -1) {
      // One or both dimensions is unknown. Follow TensorFlow behavior:
      // - If either dimension is greater than 1, we assume that the program is
      //   correct, and the other dimension will be broadcast to match it.
      // - If either dimension is 1, the other dimension is the output.
      if (*i1 > 1) {
        *iR = *i1;
      } else if (*i2 > 1) {
        *iR = *i2;
      } else if (*i1 == 1) {
        *iR = *i2;
      } else if (*i2 == 1) {
        *iR = *i1;
      } else {
        *iR = -1;
      }
    } else {
      if (*i1 == *i2 || *i2 == 1) {
        *iR = *i1;
      } else if (*i1 == 1) {
        *iR = *i2;
      } else {
        // This dimension of the two operand types is incompatible.
        return false;
      }
    }
  }

  if (dtype == DataType::invalid) {
    dtype = promoteTypes(into->elementType, from.elementType);
  }
  *into = TensorShape{dtype, resultShape};
  IVLOG(6, "  Resulting shape: " << into->str());
  return true;
}

TensorShape inferShape(llvm::ArrayRef<TensorShape> operands,
                       DataType override) {
  TensorShape ret = operands.front();
  if (override != DataType::invalid) {
    ret.elementType = override;
  }
  for (const TensorShape &operand : operands.drop_front()) {
    if (!mergeShapes(&ret, operand, override)) {
      std::stringstream ss;
      ss << "Incompatible types: (";
      for (size_t i = 0; i < operands.size(); i++) {
        if (i) {
          ss << ", ";
        }
        ss << operands[i].str();
      }
      ss << ")";
      throw std::runtime_error(ss.str());
    }
  }
  return ret;
}

//
// TensorShape
//

std::string TensorShape::str() const {
  std::stringstream ss;
  for (int64_t dim : sizes) {
    if (dim) {
      ss << dim;
    } else {
      ss << '?';
    }
    ss << 'x';
  }
  ss << util::stringifyDataType(elementType).str();
  return ss.str();
}

static size_t getByteSize(DataType dtype) {
  switch (dtype) {
  case DataType::i1:
  case DataType::si8:
  case DataType::ui8:
    return 1;
  case DataType::si16:
  case DataType::ui16:
  case DataType::bf16:
  case DataType::f16:
    return 2;
  case DataType::si32:
  case DataType::ui32:
  case DataType::f32:
    return 4;
  case DataType::si64:
  case DataType::ui64:
  case DataType::f64:
    return 8;
  default:
    break;
  }
  llvm_unreachable("Invalid DataType for getByteSize");
}

size_t TensorShape::getByteSize() const {
  size_t product = 1;
  for (size_t dim : sizes) {
    product *= dim;
  }
  return product * ast::getByteSize(elementType);
}

//
// ExprNode tree
//

std::string ExprNodeCast::str() const { return "cast"; }

TensorShape ExprNodeCast::getShape(size_t ordinal) {
  IVLOG(1, llvm::formatv("ExprNodeCast::getShape({0})", ordinal).str());
  TensorShape shape = expr->getShape();
  shape.elementType = dtype;
  return shape;
}

std::string ExprNodeConstSigned::str() const { return std::to_string(value); }

TensorShape ExprNodeConstSigned::getShape(size_t ordinal) {
  IVLOG(1, llvm::formatv("ExprNodeSigned::getShape({0})", ordinal).str());
  return TensorShape{DataType::six};
}

std::string ExprNodeConstUnsigned::str() const { return std::to_string(value); }

TensorShape ExprNodeConstUnsigned::getShape(size_t ordinal) {
  IVLOG(1, llvm::formatv("ExprNodeUnsigned::getShape({0})", ordinal).str());
  return TensorShape{DataType::uix};
}

std::string ExprNodeConstFloat::str() const { return std::to_string(value); }

TensorShape ExprNodeConstFloat::getShape(size_t ordinal) {
  IVLOG(1, llvm::formatv("ExprNodeFloat::getShape({0})", ordinal).str());
  return TensorShape{DataType::fx};
}

TensorShape ExprNodeConstTensor::getShape(size_t ordinal) {
  IVLOG(1, llvm::formatv("ExprNodeTensor::getShape({0})", ordinal).str());
  return shape;
}

std::string ExprNodeConstTensor::str() const { return "constant_tensor"; }

std::string Constraint::str() const {
  return llvm::formatv("{0} < {1}", lhs->str(), rhs->str());
}

std::string ExprNodeContraction::str() const { return "contraction"; }

TensorShape ExprNodeContraction::getShape(size_t ordinal) {
  IVLOG(1, llvm::formatv("ExprNodeContraction::getShape({0})", ordinal).str());
  DataType elementType = inferElementType(comboKind, srcs);
  TensorShape shape{elementType};
  for (const DimNodePtr &dim : sinkDims) {
    shape.sizes.push_back(dim->eval());
  }
  IVLOG(1, "  " << shape.str());
  return shape;
}

std::string ExprNodeDim::str() const { return dim->str(); }

TensorShape ExprNodeDim::getShape(size_t ordinal) {
  IVLOG(1, llvm::formatv("ExprNodeDim::getShape({0})", ordinal).str());
  return TensorShape(DataType::six);
}

std::string ExprNodeElement::str() const {
  return llvm::formatv("{0}[{1}]", expr->str(), ordinal);
}

TensorShape ExprNodeElement::getShape(size_t ordinal) {
  IVLOG(1,
        llvm::formatv("ExprNodeElement::getShape({0})", this->ordinal).str());
  return expr->getShape(this->ordinal);
}

std::string ExprNodeInput::str() const {
  if (name.size()) {
    return llvm::formatv("input<{0}, \"{1}\">", shape.str(), name);
  }
  return llvm::formatv("input<{0}>", shape.str());
}

// TensorShape ExprNodeInput::getShape() {
//   shape.dtype = in_shape.dtype;
//   shape.dims.clear();
//   for (size_t i = 0; i < in_shape.dims.size(); i++) {
//     const auto &dim = in_shape.dims[i];
//     if (auto int_expr = std::dynamic_pointer_cast<DimIntExpr>(dim.expr)) {
//       if (int_expr->value) {
//         shape.dims.push_back(dim);
//       } else {
//         shape.dims.push_back(LogicalDim{std::make_shared<DimRefExpr>(ref,
//         i)});
//       }
//     } else {
//       shape.dims.push_back(dim);
//     }
//   }
// }

TensorShape ExprNodeInput::getShape(size_t ordinal) {
  IVLOG(1, llvm::formatv("ExprNodeInput::getShape({0})", ordinal).str());
  return shape;
}

std::string ExprNodeIntrinsic::str() const {
  return llvm::formatv("{0}()", op);
}

TensorShape ExprNodeIntrinsic::getShape(size_t ordinal) {
  IVLOG(1, llvm::formatv("ExprNodeIntrinsic::getShape({0})", ordinal).str());
  llvm::SmallVector<TensorShape, 8> shapes;
  for (const ExprNodePtr &operand : operands) {
    shapes.emplace_back(operand->getShape());
  }
  auto intrinsic = IntrinsicRegistry::Instance()->resolve(op);
  if (intrinsic) {
    return intrinsic->getShape(operands, shapes, ordinal);
  }
  return inferShape(shapes);
}

std::string ExprNodeTrace::str() const {
  return llvm::formatv("trace(\"{0}\")", msg);
}

TensorShape ExprNodeTrace::getShape(size_t ordinal) {
  IVLOG(1, llvm::formatv("ExprNodeTrace::getShape({0})", ordinal).str());
  return expr->getShape();
}

//
// DimNode tree
//

std::string DimNodeLiteral::str() const { return std::to_string(value); }

int64_t DimNodeNone::eval() const {
  throw std::runtime_error("DimNodeNone cannot be evaluated");
}

std::string DimNodeOp::str() const { return getAffineOpStr(op).str(); }

int64_t DimNodeOp::eval() const {
  if (op == AffineOp::Neg) {
    if (operands.size() != 1) {
      throw std::runtime_error("Invalid number of operands in DimNodeOp");
    }
    return -operands[0]->eval();
  }
  if (operands.size() != 2) {
    throw std::runtime_error("Invalid number of operands in DimNodeOp");
  }
  int64_t lhs = operands[0]->eval();
  int64_t rhs = operands[1]->eval();
  switch (op) {
  case AffineOp::Add:
    return lhs + rhs;
  case AffineOp::Sub:
    return lhs - rhs;
  case AffineOp::Mul:
    return lhs * rhs;
  case AffineOp::Div:
    return lhs / rhs;
  case AffineOp::Max:
    return std::max(lhs, rhs);
  case AffineOp::Min:
    return std::min(lhs, rhs);
  default:
    throw std::runtime_error("Invalid AffineOp");
  }
}

std::string DimNodeRef::str() const {
  return llvm::formatv("{0}[{1}]", ref->str(), dim);
}

int64_t DimNodeRef::eval() const {
  IVLOG(1, "DimNodeRef::eval");
  return ref->getShape().sizes[dim];
}

//
// PolyNode tree
//

std::string PolyNodeDim::str() const { return dim->str(); }

std::string PolyNodeIndex::str() const { return llvm::formatv("%{0}", name); }

std::string PolyNodeLiteral::str() const { return std::to_string(value); }

std::string PolyNodeOp::str() const { return getAffineOpStr(op).str(); }

//
// VarNode tree
//

std::string VarNodeTuple::str() const {
  std::stringstream ss;
  ss << '(';
  for (auto item : llvm::enumerate(values)) {
    if (item.index()) {
      ss << ", ";
    }
    ss << item.value()->str();
  }
  ss << ')';
  return ss.str();
}

} // namespace pmlc::ast
