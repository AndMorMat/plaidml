// Copyright 2020 Intel Corporation

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "mlir/Support/TypeID.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include "pmlc/util/buffer.h"
#include "pmlc/util/enums.h"

namespace pmlc::ast {

struct TensorShape;

struct ExprNode;
struct ExprNodeCast;
struct ExprNodeConstSigned;
struct ExprNodeConstUnsigned;
struct ExprNodeConstFloat;
struct ExprNodeConstTensor;
struct ExprNodeContraction;
struct ExprNodeDim;
struct ExprNodeElement;
struct ExprNodeInput;
struct ExprNodeIntrinsic;
struct ExprNodeTrace;

struct DimNode;
struct DimNodeLiteral;
struct DimNodeNone;
struct DimNodeOp;
struct DimNodeRef;

struct PolyNode;
struct PolyNodeDim;
struct PolyNodeIndex;
struct PolyNodeLiteral;
struct PolyNodeOp;

struct VarNode;
struct VarNodeDim;
struct VarNodeExpr;
struct VarNodeFloat;
struct VarNodeInt;
struct VarNodeNone;
struct VarNodeString;
struct VarNodeTuple;

using ExprNodePtr = std::shared_ptr<ExprNode>;
using DimNodePtr = std::shared_ptr<DimNode>;
using PolyNodePtr = std::shared_ptr<PolyNode>;
using VarNodePtr = std::shared_ptr<VarNode>;

enum class AffineOp {
  Add,
  Div,
  Max,
  Min,
  Mul,
  Neg,
  Sub,
};

struct TensorShape {
  util::DataType elementType;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;

  explicit TensorShape(util::DataType elementType = util::DataType::invalid)
      : elementType(elementType) {}

  TensorShape(util::DataType elementType, llvm::ArrayRef<int64_t> sizes)
      : elementType(elementType), sizes(sizes) {}

  TensorShape(util::DataType elementType, llvm::ArrayRef<int64_t> sizes,
              llvm::ArrayRef<int64_t> strides)
      : elementType(elementType), sizes(sizes), strides(strides) {}

  std::string str() const;
  size_t getRank() const { return sizes.size(); }
  size_t getByteSize() const;
};

util::DataType inferElementType(llvm::ArrayRef<TensorShape> shapes);

TensorShape inferShape(llvm::ArrayRef<TensorShape> operands,
                       util::DataType override = util::DataType::invalid);

//
// Base AST Node
//

template <typename ConcreteT, typename BaseT>
struct NodeBase : BaseT, std::enable_shared_from_this<ConcreteT> {
  using BaseT::BaseT;

  mlir::TypeID getTypeID() const final {
    return mlir::TypeID::get<ConcreteT>();
  }

  /// Provide an implementation of 'classof' that compares the type id of the
  /// provided value with that of the concerete type.
  static bool classof(const BaseT *val) {
    return val->getTypeID() == mlir::TypeID::get<ConcreteT>();
  }

  std::shared_ptr<ConcreteT> as_ptr() { return this->shared_from_this(); }
};

//
// ExprNode tree
//

struct ExprNode {
  std::string name;

  explicit ExprNode(llvm::StringRef name = "") : name(name) {}
  virtual ~ExprNode() = default;
  virtual mlir::TypeID getTypeID() const = 0;
  virtual std::string str() const = 0;
  virtual TensorShape getShape(size_t ordinal = 0) = 0;
};

struct ExprNodeInput : NodeBase<ExprNodeInput, ExprNode> {
  using Base = NodeBase<ExprNodeInput, ExprNode>;

  TensorShape shape;

  explicit ExprNodeInput(const TensorShape &shape, llvm::StringRef name = "")
      : Base(name), shape(shape) {}
  std::string str() const final;
  TensorShape getShape(size_t ordinal = 0) final;
};

struct ExprNodeCast : NodeBase<ExprNodeCast, ExprNode> {
  util::DataType dtype;
  ExprNodePtr expr;

  explicit ExprNodeCast(util::DataType dtype, const ExprNodePtr &expr)
      : dtype(dtype), expr(expr) {}
  std::string str() const final;
  TensorShape getShape(size_t ordinal = 0) final;
};

struct ExprNodeConstSigned : NodeBase<ExprNodeConstSigned, ExprNode> {
  int64_t value;

  explicit ExprNodeConstSigned(int64_t value) : value(value) {}
  std::string str() const final;
  TensorShape getShape(size_t ordinal = 0) final;
};

struct ExprNodeConstUnsigned : NodeBase<ExprNodeConstUnsigned, ExprNode> {
  uint64_t value;

  explicit ExprNodeConstUnsigned(uint64_t value) : value(value) {}
  std::string str() const final;
  TensorShape getShape(size_t ordinal = 0) final;
};

struct ExprNodeConstFloat : NodeBase<ExprNodeConstFloat, ExprNode> {
  double value;

  explicit ExprNodeConstFloat(double value) : value(value) {}
  std::string str() const final;
  TensorShape getShape(size_t ordinal = 0) final;
};

struct ExprNodeConstTensor : NodeBase<ExprNodeConstTensor, ExprNode> {
  using Base = NodeBase<ExprNodeConstTensor, ExprNode>;

  TensorShape shape;
  pmlc::util::BufferPtr buffer;

  explicit ExprNodeConstTensor(const TensorShape &shape,
                               const pmlc::util::BufferPtr &buffer,
                               llvm::StringRef name = "")
      : Base(name), shape(shape), buffer(buffer) {}
  std::string str() const final;
  TensorShape getShape(size_t ordinal = 0) final;
};

struct PolyMap {
  ExprNodePtr ref;
  std::vector<PolyNodePtr> idxs;
};

struct Constraint {
  PolyNodePtr lhs;
  DimNodePtr rhs;

  Constraint(const PolyNodePtr &lhs, const DimNodePtr &rhs)
      : lhs(lhs), rhs(rhs) {}
  std::string str() const;
};

struct ExprNodeContraction : NodeBase<ExprNodeContraction, ExprNode> {
  using Base = NodeBase<ExprNodeContraction, ExprNode>;

  pmlc::util::AggregationKind aggKind;
  pmlc::util::CombinationKind comboKind;
  std::vector<DimNodePtr> sinkDims;
  std::vector<PolyNodePtr> sinkIdxs;
  std::vector<PolyMap> srcs;
  std::vector<Constraint> constraints;
  bool simplify = true;
  ExprNodePtr init;

  explicit ExprNodeContraction(llvm::StringRef name = "") : Base(name) {}
  std::string str() const final;
  TensorShape getShape(size_t ordinal = 0) final;
};

struct ExprNodeDim : NodeBase<ExprNodeDim, ExprNode> {
  DimNodePtr dim;

  explicit ExprNodeDim(const DimNodePtr &dim) : dim(dim) {}
  std::string str() const final;
  TensorShape getShape(size_t ordinal = 0) final;
};

struct ExprNodeElement : NodeBase<ExprNodeElement, ExprNode> {
  ExprNodePtr expr;
  size_t ordinal;

  ExprNodeElement(const ExprNodePtr &expr, size_t ordinal)
      : expr(expr), ordinal(ordinal) {}
  std::string str() const final;
  TensorShape getShape(size_t ordinal) final;
};

struct ExprNodeIntrinsic : NodeBase<ExprNodeIntrinsic, ExprNode> {
  std::string op;
  std::vector<ExprNodePtr> operands;

  ExprNodeIntrinsic(llvm::StringRef op, llvm::ArrayRef<ExprNodePtr> operands)
      : op(op), operands(operands) {}
  std::string str() const final;
  TensorShape getShape(size_t ordinal = 0) final;
};

struct ExprNodeTrace : NodeBase<ExprNodeTrace, ExprNode> {
  ExprNodePtr expr;
  std::string msg;

  ExprNodeTrace(const ExprNodePtr &expr, llvm::StringRef msg)
      : expr(expr), msg(msg) {}
  std::string str() const final;
  TensorShape getShape(size_t ordinal = 0) final;
};

//
// DimNode tree
//

struct DimNode {
  virtual ~DimNode() = default;
  virtual mlir::TypeID getTypeID() const = 0;
  virtual std::string str() const = 0;
  virtual int64_t eval() const = 0;
};

struct DimNodeLiteral : NodeBase<DimNodeLiteral, DimNode> {
  int64_t value;

  explicit DimNodeLiteral(int64_t value) : value(value) {}
  std::string str() const final;
  int64_t eval() const final { return value; }
};

struct DimNodeNone : NodeBase<DimNodeNone, DimNode> {
  std::string str() const final { return "none"; }
  int64_t eval() const final;
};

struct DimNodeOp : NodeBase<DimNodeOp, DimNode> {
  AffineOp op;
  std::vector<DimNodePtr> operands;

  DimNodeOp(AffineOp op, llvm::ArrayRef<DimNodePtr> operands)
      : op(op), operands(operands) {}
  std::string str() const final;
  int64_t eval() const final;
};

struct DimNodeRef : NodeBase<DimNodeRef, DimNode> {
  ExprNodePtr ref;
  size_t dim;

  DimNodeRef(const ExprNodePtr &ref, size_t dim) : ref(ref), dim(dim) {}
  std::string str() const final;
  int64_t eval() const final;
};

//
// PolyNode tree
//

struct PolyNode {
  virtual ~PolyNode() = default;
  virtual mlir::TypeID getTypeID() const = 0;
  virtual std::string str() const = 0;
};

struct PolyNodeDim : NodeBase<PolyNodeDim, PolyNode> {
  DimNodePtr dim;

  explicit PolyNodeDim(const DimNodePtr &dim) : dim(dim) {}
  std::string str() const final;
};

struct PolyNodeIndex : NodeBase<PolyNodeIndex, PolyNode> {
  std::string name;

  explicit PolyNodeIndex(llvm::StringRef name = "") : name(name) {}
  std::string str() const final;
};

struct PolyNodeLiteral : NodeBase<PolyNodeLiteral, PolyNode> {
  int64_t value;

  explicit PolyNodeLiteral(int64_t value) : value(value) {}
  std::string str() const final;
};

struct PolyNodeOp : NodeBase<PolyNodeOp, PolyNode> {
  AffineOp op;
  std::vector<PolyNodePtr> operands;

  PolyNodeOp(AffineOp op, llvm::ArrayRef<PolyNodePtr> operands)
      : op(op), operands(operands) {}
  std::string str() const final;
};

//
// VarNode tree
//

struct VarNode {
  virtual ~VarNode() = default;
  virtual mlir::TypeID getTypeID() const = 0;
  virtual std::string str() const = 0;
};

struct VarNodeDim : NodeBase<VarNodeDim, VarNode> {
  DimNodePtr value;
  explicit VarNodeDim(DimNodePtr value) : value(value) {}
  std::string str() const final { return value->str(); }
};

struct VarNodeExpr : NodeBase<VarNodeExpr, VarNode> {
  ExprNodePtr value;
  explicit VarNodeExpr(ExprNodePtr value) : value(value) {}
  std::string str() const final { return value->str(); }
};

struct VarNodeFloat : NodeBase<VarNodeFloat, VarNode> {
  double value;
  explicit VarNodeFloat(double value) : value(value) {}
  std::string str() const final { return std::to_string(value); };
};

struct VarNodeInt : NodeBase<VarNodeInt, VarNode> {
  int64_t value;
  explicit VarNodeInt(int64_t value) : value(value) {}
  std::string str() const final { return std::to_string(value); };
};

struct VarNodeNone : NodeBase<VarNodeNone, VarNode> {
  std::string str() const final { return "none"; };
};

struct VarNodeString : NodeBase<VarNodeString, VarNode> {
  std::string value;
  explicit VarNodeString(llvm::StringRef value) : value(value) {}
  std::string str() const final { return value; };
};

struct VarNodeTuple : NodeBase<VarNodeTuple, VarNode> {
  std::vector<VarNodePtr> values;
  std::string str() const final;
};

} // namespace pmlc::ast
