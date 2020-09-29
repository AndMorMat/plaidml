// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <memory>
#include <vector>

namespace pmlc::util {

class Buffer;
using BufferPtr = std::shared_ptr<Buffer>;

// Buffer represents a buffer residing on some Platform.
class Buffer {
public:
  virtual ~Buffer() {}
  virtual size_t size() const = 0;
  virtual char *data() = 0;
  virtual BufferPtr clone() = 0;
};

// A simple buffer backed by a std::vector
class SimpleBuffer : public Buffer,
                     public std::enable_shared_from_this<SimpleBuffer> {
public:
  explicit SimpleBuffer(size_t size) : data_(size) {}

  explicit SimpleBuffer(const std::vector<char> &data) : data_(data) {}

  size_t size() const final { return data_.size(); }

  char *data() final { return data_.data(); }

  BufferPtr clone() final { return std::make_shared<SimpleBuffer>(data_); }

private:
  std::vector<char> data_;
};

// An adopted buffer owned by the user.
class AdoptedBuffer : public Buffer,
                      public std::enable_shared_from_this<AdoptedBuffer> {
public:
  AdoptedBuffer(size_t size, char *data) : size_(size), data_(data) {}

  size_t size() const final { return size_; }

  char *data() final { return data_; }

  BufferPtr clone() final {
    return std::make_shared<AdoptedBuffer>(size_, data_);
  }

private:
  size_t size_;
  char *data_;
};

} // namespace pmlc::util
