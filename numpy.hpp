#ifndef __CXX_NUMPY_H__
#define __CXX_NUMPY_H__
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <boost/python.hpp>
#include <boost/python/object.hpp>
//#include <boost/mpl/int.hpp>
#include "assert.h"

namespace boost { namespace python { namespace numeric {

namespace mw_py_impl
{
  // see here http://stackoverflow.com/questions/16454987/c-generate-a-compile-error-if-user-tries-to-instantiate-a-class-template-wiho
  template<class T>
  class incomplete;
}

  
void importNumpyAndRegisterTypes();
  
enum { MAX_DIM = 32 };
  
// see here http://stackoverflow.com/questions/16454987/c-generate-a-compile-error-if-user-tries-to-instantiate-a-class-template-wiho
template<class T>
int getItemtype()
{
  enum { S = sizeof(mw_py_impl::incomplete<T>) }; // will generate an error
  return -1;
}

#define NUMPY_HPP_FWD_DECLARE_GET_ITEMTYPE(T)\
  template<> int getItemtype<T>();
  
NUMPY_HPP_FWD_DECLARE_GET_ITEMTYPE(PyObject*)
NUMPY_HPP_FWD_DECLARE_GET_ITEMTYPE(float)
NUMPY_HPP_FWD_DECLARE_GET_ITEMTYPE(double)
NUMPY_HPP_FWD_DECLARE_GET_ITEMTYPE(int)
NUMPY_HPP_FWD_DECLARE_GET_ITEMTYPE(long)
NUMPY_HPP_FWD_DECLARE_GET_ITEMTYPE(long long)
NUMPY_HPP_FWD_DECLARE_GET_ITEMTYPE(short)
NUMPY_HPP_FWD_DECLARE_GET_ITEMTYPE(char)
NUMPY_HPP_FWD_DECLARE_GET_ITEMTYPE(bool)
NUMPY_HPP_FWD_DECLARE_GET_ITEMTYPE(unsigned int)
NUMPY_HPP_FWD_DECLARE_GET_ITEMTYPE(unsigned long)
NUMPY_HPP_FWD_DECLARE_GET_ITEMTYPE(unsigned short)
NUMPY_HPP_FWD_DECLARE_GET_ITEMTYPE(unsigned char)
NUMPY_HPP_FWD_DECLARE_GET_ITEMTYPE(unsigned long long)
#undef NUMPY_HPP_FWD_DECLARE_GET_ITEMTYPE


array zeros(int rank, const int *dims, int type);
array empty(int rank, const int *dims, int type);

int getItemtype(const array &arr);

template<class T>
bool isCompatibleType(int id);

class arraytbase
{ 
protected:
  array obj; // the object
  int r, itemsize_, itemtype_; // rank
  int m[MAX_DIM],s[MAX_DIM]; // strides and size
  unsigned char* bytes_; // data
  bool is_ok_;

public:
  arraytbase();
  arraytbase(object const &a);
  
  const int* shape() const { return s; }
  int itemtype() const { return itemtype_; }
  int itemsize() const { return itemsize_; }
  int rank() const { return r; }
  const int* strides() const { return m; } // in number of bytes, *not* number of items!
  bool isWriteable() const;
  bool isCContiguous() const;
  bool isFContiguous() const;
  const array& getObject() const { return obj; }
  
  inline int offset(int x) const
  {
    assert((unsigned int)x<s[0]);
    return x*m[0];
  }
  
  inline int offset(int x, int y) const
  {
    assert((unsigned int)y<s[1]);
    return offset(x) + y*m[1];
  }
  
  inline int offset(int x, int y, int z) const
  {
    assert((unsigned int)z<s[2]);
    return offset(x,y) + z*m[2];
  }
  
  inline int offset(int x,int y, int z, int w) const
  {
    assert((unsigned int)w<s[3]);
    return offset(x,y,z) + w*m[3];
  }

  inline int offset(const int *c) const
  {
    int q = 0;
    for(int i=0; i<r; ++i)
    {
      assert((unsigned int)(c[i])<s[i]);
      q += c[i]*m[i];
    }
    return q;
  }

  unsigned char* bytes()
  {
    return bytes_;
  }
};


template<class T>
class arrayt : public arraytbase
{
public:
  arrayt() : arraytbase() {}
  arrayt(arraytbase const &a) : arraytbase(a.getObject()) 
  {
    if (sizeof(T) != arraytbase::itemsize())
      throw std::invalid_argument("arrayt: array itemsize does not match template argument");
  }
  arrayt(object const &a) : arraytbase(a)
  {
    if (sizeof(T) != arraytbase::itemsize())
      throw std::invalid_argument("arrayt: array itemsize does not match template argument");
  }

  void init(object const &a_)
  {
    this->~arrayt<T>();
    new (this) arrayt<T>(a_);
  }

  T& operator()(int x)
  {
    return *((T*)(arraytbase::bytes_+offset(x)));
  }
  
  T& operator()(int x, int y)
  {
    return *((T*)(arraytbase::bytes_+offset(x,y)));
  }
  
  T& operator()(int x, int y, int z)
  {
    return *((T*)(arraytbase::bytes_+offset(x,y,z)));
  }
  
  T& operator()(int x, int y, int z, int w)
  {
    return *((T*)(arraytbase::bytes_+offset(x,y,z,w)));
  }
  
  T& operator()(int *c)
  {
    return *((T*)(arraytbase::bytes_+offset(c)));
  }
  
  T& operator[](int i)
  {
    return *((T*)(arraytbase::bytes_+offset(i, 0, 0, 0)));
  }

  T* data() { return reinterpret_cast<T*>(bytes()); }
  
  
  T operator()(int x) const
  {
    return *((T*)(arraytbase::bytes_+offset(x)));
  }
  
  T operator()(int x, int y) const
  {
    return *((T*)(arraytbase::bytes_+offset(x,y)));
  }
  
  T operator()(int x, int y, int z) const
  {
    return *((T*)(arraytbase::bytes_+offset(x,y,z)));
  }
  
  T operator()(int x, int y, int z, int w) const
  {
    return *((T*)(arraytbase::bytes_+offset(x,y,z,w)));
  }
  
  T operator()(int *c) const
  {
    return *((T*)(arraytbase::bytes_+offset(c)));
  }
  
  T operator[](int i) const
  {
    return *((T*)(arraytbase::bytes_+offset(i, 0, 0, 0)));
  }
  
  const T* data() const { return reinterpret_cast<T*>(bytes()); }
};


namespace mw_py_impl
{

template<class T>
inline void gridded_data_ccons(T* dst, const T* src, const int* dims, const int *dst_strides, const int *src_strides, boost::mpl::int_<0>)
{
  for (int i=0; i<dims[0]; ++i)
  {
    new (dst) T(*src);
    dst += dst_strides[0];
    src += src_strides[0];
  }
}

template<class T, int dim>
inline void gridded_data_ccons(T* dst, const T* src, const int* dims, const int *dst_strides, const int *src_strides, boost::mpl::int_<dim>)
{
  for (int i=0; i<dims[dim]; ++i)
  {
    gridded_data_ccons(dst, src, dims, dst_strides, src_strides, boost::mpl::int_<dim-1>());
    dst += dst_strides[dim];
    src += src_strides[dim];
  }
}

}


/*
 * WARNING: strides given in number of items, *not in bytes*
 */
template<class T, int rank>
static array copy(const int* dims, const T* src, const int *strides)
{
  int item_type = getItemtype<T>();
  arrayt<T> arr(empty(rank, dims, item_type));

  int arr_strides[MAX_DIM];
  for (int i=0; i<rank; ++i) arr_strides[i] = arr.strides()[i]/arr.itemsize();
  
  T* dst = arr.data();
  mw_py_impl::gridded_data_ccons<T>(dst, src, dims, arr_strides, strides, boost::mpl::int_<rank-1>());
  
  return arr.getObject();
}

template<class T, int rank>
static void copy(T* dst, const int *dims, const int* strides, const array &pyarr)
{
  arrayt<T> arr(pyarr);

  int arr_strides[MAX_DIM];
  for (int i=0; i<rank; ++i) {
    assert(arr.shape()[i] == dims[i]);
    arr_strides[i] = arr.strides()[i]/arr.itemsize();
  }
  
  const T* src = arr.data();
  mw_py_impl::gridded_data_ccons<T>(dst, src, arr.shape(), strides, arr_strides, boost::mpl::int_<rank-1>());
}


} } }

#endif