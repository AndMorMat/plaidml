// Copyright 2019, Intel Corporation

#include "pmlc/dialect/stripe/util.h"

#include <limits>

#include "mlir/IR/Builders.h"

#include "base/util/logging.h"

using mlir::NamedAttribute;
using mlir::OpBuilder;
using pmlc::dialect::stripe::ParallelForOp;
using pmlc::dialect::stripe::TerminateOp;

namespace pmlc {
namespace dialect {
namespace stripe {

void createMainParallelFor(mlir::FuncOp funcOp) {
  auto& region = funcOp.getBody();
  OpBuilder builder(region);
  auto src = &region.front();
  auto it = src->begin();
  auto forOp = builder.create<ParallelForOp>(funcOp.getLoc(), builder.getI64ArrayAttr({}));
  auto attrs = llvm::SmallVector<NamedAttribute, 1>{
      {builder.getIdentifier("main"), builder.getUnitAttr()},
  };
  forOp.setAttr(dialect::stripe::Dialect::getStripeAttrsName(), builder.getDictionaryAttr(attrs));
  forOp.setAttr("name", builder.getStringAttr("main"));
  auto block = builder.createBlock(&forOp.inner());
  auto& dst = block->getOperations();
  dst.splice(dst.end(), src->getOperations(), it, src->end());

  builder.setInsertionPointToEnd(src);
  builder.create<TerminateOp>(funcOp.getLoc());
}

bool hasAttr(Operation* op, StringRef attr) {
  std::set<std::string> op_attrs_set;
  DictionaryAttr dict_attr = op->getAttrOfType<DictionaryAttr>(Dialect::getStripeAttrsName());
  if (!dict_attr) {
    return false;
  }
  ArrayRef<NamedAttribute> op_attrs = dict_attr.getValue();
  for (const auto& [key, value] : op_attrs) {
    auto name = key.strref();
    op_attrs_set.insert(name);
  }
  return op_attrs_set.find(attr) != op_attrs_set.end();
}

bool hasAttrs(Operation* op, const std::set<std::string>& attrs) {
  std::set<std::string> op_attrs_set;
  DictionaryAttr dict_attr = op->getAttrOfType<DictionaryAttr>(Dialect::getStripeAttrsName());
  if (!dict_attr) {
    return false;
  }
  ArrayRef<NamedAttribute> op_attrs = dict_attr.getValue();
  for (const auto& [key, value] : op_attrs) {
    auto name = key.strref();
    op_attrs_set.insert(name);
  }
  for (const auto& attr : attrs) {
    if (op_attrs_set.find(attr) == op_attrs_set.end()) {
      return false;
    }
  }
  return true;
}

DictionaryAttr addAttrInDictionary(DictionaryAttr old_dict, OpBuilder builder, NamedAttribute elem) {
  llvm::SmallVector<NamedAttribute, 8> new_array;
  if (old_dict) {
    ArrayRef<NamedAttribute> old_array = old_dict.getValue();
    new_array.insert(new_array.begin(), old_array.begin(), old_array.end());
  }
  new_array.emplace_back(elem);
  return builder.getDictionaryAttr(new_array);
}

ArrayAttr addAttrInArray(ArrayAttr old_array, OpBuilder builder, Attribute elem) {
  llvm::SmallVector<Attribute, 8> new_array;
  if (old_array) {
    ArrayRef<Attribute> elements = old_array.getValue();
    new_array.insert(new_array.begin(), elements.begin(), elements.end());
  }
  new_array.emplace_back(elem);
  ArrayRef new_array_ref(new_array.begin(), new_array.end());
  return builder.getArrayAttr(new_array_ref);
}

DictionaryAttr replaceAttrInDictionary(DictionaryAttr old_dict, OpBuilder builder, unsigned n, NamedAttribute elem) {
  llvm::SmallVector<NamedAttribute, 8> new_array;
  if (old_dict) {
    ArrayRef<NamedAttribute> old_array = old_dict.getValue();
    new_array.insert(new_array.begin(), old_array.begin(), old_array.end());
  }
  while (n >= new_array.size()) {
    new_array.emplace_back(builder.getIdentifier(""), builder.getUnitAttr());
  }
  new_array[n] = elem;
  return builder.getDictionaryAttr(new_array);
}

ArrayAttr replaceAttrInArray(ArrayAttr old_array, OpBuilder builder, unsigned n, Attribute elem) {
  llvm::SmallVector<Attribute, 8> new_array;
  if (old_array) {
    ArrayRef<Attribute> elements = old_array.getValue();
    new_array.insert(new_array.begin(), elements.begin(), elements.end());
  }
  while (n >= new_array.size()) {
    new_array.emplace_back(builder.getUnitAttr());
  }
  new_array[n] = elem;
  ArrayRef new_array_ref(new_array.begin(), new_array.end());
  return builder.getArrayAttr(new_array_ref);
}

void setOpAttrUnit(Operation* op, OpBuilder builder, StringRef attr_name) {
  if (!op) {
    throw std::runtime_error("setUnitAttr: op is null");
  }
  auto old_attrs_dict = op->getAttrOfType<DictionaryAttr>(Dialect::getStripeAttrsName());
  NamedAttribute new_elem = {builder.getIdentifier(attr_name), builder.getUnitAttr()};
  auto new_attrs_dict = addAttrInDictionary(old_attrs_dict, builder, new_elem);
  op->setAttr(Dialect::getStripeAttrsName(), new_attrs_dict);
}

void setIdxAttrUnit(ParallelForOp op, StringRef target_idx, StringRef attr_name) {
  auto idx_names = op.getAttrOfType<ArrayAttr>("idx_names");
  if (!idx_names) {
    return;
  }
  auto old_idx_attrs = op.getAttrOfType<ArrayAttr>("idx_attrs");
  ArrayAttr new_idx_attrs;
  auto builder = op.getBodyBuilder();
  for (unsigned i = 0; i < op.ranges().size(); i++) {
    StringRef idx_name;
    if (i < idx_names.size()) {
      if (auto str_attr = idx_names.getValue()[i].dyn_cast<StringAttr>()) {
        idx_name = str_attr.getValue();
      }
    }
    if (idx_name == target_idx) {
      DictionaryAttr old_dict;
      if (old_idx_attrs && i < old_idx_attrs.size()) {
        old_dict = old_idx_attrs.getValue()[i].dyn_cast<DictionaryAttr>();
      }
      NamedAttribute new_elem = {builder.getIdentifier(attr_name), builder.getUnitAttr()};
      DictionaryAttr new_dict = addAttrInDictionary(old_dict, builder, new_elem);
      new_idx_attrs = replaceAttrInArray(old_idx_attrs, builder, i, new_dict);
      break;
    }
  }
  op.setAttr("idx_attrs", new_idx_attrs);
}

int64_t idxRange(BlockArgument idx) {
  auto pf = mlir::cast<ParallelForOp>(idx.getOwner()->getParentOp());
  return pf.getRange(idx.getArgNumber());
}

StringRef idxName(BlockArgument idx) {
  auto op = mlir::cast<ParallelForOp>(idx.getOwner()->getParentOp());
  auto idxNames = op.getAttrOfType<ArrayAttr>("idx_names");
  auto argNumber = idx.getArgNumber();
  if (idxNames && argNumber < idxNames.size()) {
    auto argAttr = idxNames.getValue()[argNumber];
    if (auto strAttr = argAttr.dyn_cast<StringAttr>()) {
      return strAttr.getValue();
    }
  }
  return "";
}

std::pair<StringRef, unsigned> getSingleIndex(ParallelForOp op, unsigned n) {
  auto names = op.getAttrOfType<ArrayAttr>("idx_names");
  auto ranges = op.ranges();
  auto idx_name = StringAttr::get("", op.getContext());
  if (names && n < names.size()) {
    if (auto str_attr = names.getValue()[n].dyn_cast<StringAttr>()) {
      idx_name = str_attr;
    }
  }
  unsigned range = ranges.getValue()[n].cast<IntegerAttr>().getInt();
  return std::make_pair(idx_name.getValue(), range);
}

void getAllIndex(ParallelForOp op, llvm::SmallVectorImpl<std::pair<StringRef, unsigned>>* idxs) {
  auto names = op.getAttrOfType<ArrayAttr>("idx_names");
  auto ranges = op.ranges();
  for (unsigned i = 0; i < op.ranges().size(); i++) {
    auto idx_name = StringAttr::get("", op.getContext());
    if (names && i < names.size()) {
      if (auto str_attr = names.getValue()[i].dyn_cast<StringAttr>()) {
        idx_name = str_attr;
      }
    }
    unsigned range = ranges.getValue()[i].cast<IntegerAttr>().getInt();
    idxs->push_back(std::make_pair(idx_name.getValue(), range));
  }
}

TensorType baseType(Value value) {
  auto cur_value = value;
  do {
    if (auto def = cur_value.getDefiningOp()) {
      if (auto aop = mlir::dyn_cast<AllocateOp>(def)) {
        return aop.layout();
      }
      auto rop = mlir::dyn_cast<RefineOp>(def);
      cur_value = rop.in();
    } else if (auto arg = cur_value.dyn_cast<BlockArgument>()) {
      auto parentOp = arg->getOwner()->getParentOp();
      auto funcOp = mlir::dyn_cast<mlir::FuncOp>(parentOp);
      if (!funcOp) {
        throw std::runtime_error("Invalid tensor value: block argument not contained by FuncOp");
      }
      auto attrName = stripe::Dialect::getDialectAttrName("layout");
      auto attr = funcOp.getArgAttrOfType<mlir::TypeAttr>(arg->getArgNumber(), attrName);
      assert(attr && "Expected 'layout' attribute in TensorRefType function argument");
      return attr.getValue().cast<TensorType>();
    } else {
      throw std::runtime_error("Invalid tensor value");
    }
  } while (cur_value);
  throw std::runtime_error("Base type not found for the operation.");
}

void strideOneIdxs(Value value, llvm::SmallVectorImpl<BlockArgument>* idxs) {
  auto ref_op = mlir::dyn_cast<RefineOp>(value.getDefiningOp());
  TensorType base_type = baseType(value);
  auto shape = base_type.getShape();
  for (unsigned i = 0; i < shape.size(); i++) {
    if (shape[i].stride != 1) {
      continue;
    }
    auto access = AffinePolynomial(ref_op.getOffset(i));
    for (auto [arg, scale] : access.terms) {
      if (scale == 1) {
        idxs->push_back(arg);
      }
    }
  }
}

StringRef tensorName(Value tensor) {
  if (auto op = tensor->getDefiningOp()) {
    auto nameAttr = op->getAttrOfType<StringAttr>("name");
    if (nameAttr) {
      return nameAttr.getValue();
    }
  }
  return StringRef();
}

DataType tensorElementType(Value tensor) {
  auto tensor_type = tensor->getType().cast<TensorRefType>();
  auto elt_type = tensor_type.getElementType().cast<eltwise::ScalarType>();
  return elt_type.type();
}

int64_t IntegerMax(DataType type) {
  switch (type) {
    case DataType::INT8:
      return std::numeric_limits<int8_t>::max();
    case DataType::INT16:
      return std::numeric_limits<int16_t>::max();
    case DataType::INT32:
      return std::numeric_limits<int32_t>::max();
    case DataType::INT64:
      return std::numeric_limits<int64_t>::max();
    default:
      return 0;
  }
  llvm_unreachable("Unreachable code");
}

int64_t IntegerMin(DataType type) {
  switch (type) {
    case DataType::INT8:
      return std::numeric_limits<int8_t>::lowest();
    case DataType::INT16:
      return std::numeric_limits<int16_t>::lowest();
    case DataType::INT32:
      return std::numeric_limits<int32_t>::lowest();
    case DataType::INT64:
      return std::numeric_limits<int64_t>::lowest();
    default:
      return 0;
  }
  llvm_unreachable("Unreachable code");
}

double FloatMax(DataType type) {
  switch (type) {
    case DataType::FLOAT16:
      throw std::runtime_error("Unsupported type for FloatMax");
    case DataType::FLOAT32:
      return std::numeric_limits<float>::max();
    case DataType::FLOAT64:
      return std::numeric_limits<double>::max();
    default:
      return 0;
  }
  llvm_unreachable("Unreachable code");
}

double FloatMin(DataType type) {
  switch (type) {
    case DataType::FLOAT16:
      throw std::runtime_error("Unsupported type for FloatMin");
    case DataType::FLOAT32:
      return std::numeric_limits<float>::lowest();
    case DataType::FLOAT64:
      return std::numeric_limits<double>::lowest();
    default:
      return 0;
  }
  llvm_unreachable("Unreachable code");
}

eltwise::ScalarConstantOp createConstOp(OpBuilder* builder, DataType type, int64_t ivalue, double fvalue) {
  auto unknownLoc = builder->getUnknownLoc();
  auto scalarType = ScalarType::get(builder->getContext(), type);
  if (is_float(type)) {
    return builder->create<eltwise::ScalarConstantOp>(unknownLoc, scalarType, fvalue);
  }
  return builder->create<eltwise::ScalarConstantOp>(unknownLoc, scalarType, ivalue);
}

eltwise::ScalarConstantOp initialValue(  //
    OpBuilder* builder,                  //
    DataType type,                       //
    AggregationKind agg,                 //
    StringRef var_name) {
  eltwise::ScalarConstantOp op;
  switch (agg) {
    case AggregationKind::assign:
      return eltwise::ScalarConstantOp();
    case AggregationKind::add:
      op = createConstOp(builder, type, 0, 0.0);
      break;
    case AggregationKind::mul:
      op = createConstOp(builder, type, 1, 1.0);
      break;
    case AggregationKind::max:
      op = createConstOp(builder, type, IntegerMin(type), FloatMin(type));
      break;
    case AggregationKind::min:
      op = createConstOp(builder, type, IntegerMax(type), FloatMax(type));
      break;
    default:
      throw std::runtime_error("Unsupported aggregation op.");
  }
  op.setAttr("scalar_name", builder->getStringAttr(var_name == "" ? "cst" : var_name));
  return op;
}

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
