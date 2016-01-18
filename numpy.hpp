#ifndef __CXX_NUMPY_H__
#define __CXX_NUMPY_H__
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <boost/python.hpp>
#include <boost/python/object.hpp>
//#include <boost/mpl/int.hpp>
#include "assert.h"

struct tagPyArrayObject;
typedef tagPyArrayObject PyArrayObject;

namespace boost { namespace python { 
  
/** 
 * @brief Everything of this library can be found in this namespace where also boost's array class is defined.
*/
namespace numeric {
  
/**
@brief Some implementation details are in here.
*/
namespace mw_py_impl
{
  // see here http://stackoverflow.com/questions/16454987/c-generate-a-compile-error-if-user-tries-to-instantiate-a-class-template-wiho
  template<class T>
  class incomplete;
};


/** @brief Equal to NPY_MAX_DIMS. Provided so that the numpy headers don't have to be included.
*/
enum {
  MAX_DIM = 32
};


/** @brief Synonymous for Py_ssize_t
*/
typedef Py_ssize_t ssize_t;


/** @brief Initializes things.

It calls
> import_array1();

> py::numeric::array::set_module_and_type("numpy", "ndarray");
And it registers several converter routines in boost python 
for automatic conversion of <arrayt>, <arraytbase> and scalar types.
*/
void importNumpyAndRegisterTypes();


/** @brief Obtain numpy's data type number from template parameter T.

By default, basic types are mapped to NPY_FLOAT32, NPY_FLOAT64, NPY_INT,
NPY_LONG, NPY_LONGLONG, NPY_SHORT, NPY_BYTE, NPY_BOOL, NPY_UINT, NPY_ULONG,
NPY_USHORT, NPY_UBYTE, NPY_ULONGLONG. PyObject* is mapped to NPY_OBJECT.

@sa getItemtype(const array &arr);
*/
template<class T>
int getItemtype()
{
// see here http://stackoverflow.com/questions/16454987/c-generate-a-compile-error-if-user-tries-to-instantiate-a-class-template-wiho
  enum { S = sizeof(mw_py_impl::incomplete<T>) }; // will generate an error for types for which getItemtype was not implemented!
  return -1;
}


/** @cond */
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
/** @endcond */

/** @brief Creates a boost::python::numeric::array filled with zeros.

The type parameter specifies the numpy data type number. 
You can use getItemtype<T>() to obtain one.

*/
py::object zeros(int rank, const Py_ssize_t *dims, int type);

/** @brief boost::python::numeric::array with uninitialized memory.

The type parameter specifies the numpy data type number. 
You can use getItemtype<T>() to obtain one.
*/
py::object empty(int rank, const Py_ssize_t *dims, int type);

/**
@brief Obtain numpy's data type number from an boost-python object which has to hold a ndarry. 

Returns the result of PyArray_TYPE().

@sa getItemtype<T>()
*/
int getItemtype(const py::object &arr);

/**
@brief Determines if a numpy data type identified by the type number id is binary compatible with the c-type given by T.

This function is specialized for basic number types: various sizes of integers, and float and double.
This is a many to one correspondence. For instance depending on the platform, numpy types 'int' and 'long' which have different type numbers are both 8 bytes in length and therefore both correspond to the long type in c++!

@sa getItemtype<T>()
*/
template<class T>
bool isCompatibleType(int id);

/**
@brief This is the base class from which type specific variants are derived.

It stores only the object pointer and an instance of boost::python::numeric::array 
for reference counting. Therefore the referenced object can be changed during the lifetime
of an arraytbase instance. An instance of this class can be extracted from a boost::python object _obj_
by _extract<arraytbase>(obj)_ or by automatic conversion by boost::pythons function wrapping facilities.

Members provided access to basic information that don't require casting to a specific element type.

\sa arrayt
*/
class arraytbase
{ 
protected:
  array obj; // the object
  PyArrayObject* objptr;
 
  arraytbase(object const &a, int typesize);
  void construct(object const &a, int typesize);

public:
  /* 
   */
  arraytbase(object const &a = object());
  typedef Py_ssize_t ssize_t;
  
  const ssize_t* shape() const;
  const ssize_t* dims() const { return shape(); }
  const ssize_t* strides() const; ///< Returned in number of bytes.
  int itemtype() const;
  int itemsize() const;
  int rank() const;
  int ndim() const { return rank(); }
  bool isWriteable() const;
  bool isCContiguous() const;
  bool isFContiguous() const;
  const array& getObject() const { return obj; }

#define NUMPY_HPP_OFFSET1 \
  x*m[0]
#define NUMPY_HPP_OFFSET2 \
  NUMPY_HPP_OFFSET1 + y*m[1]
#define NUMPY_HPP_OFFSET3 \
  NUMPY_HPP_OFFSET2 + z*m[2]
#define NUMPY_HPP_OFFSET4 \
  NUMPY_HPP_OFFSET3 + w*m[3]
  
  /// Returned in number of bytes.
  /** Returns the position from the given integer coordinates of an element 
   *  in the memory block pointed to by _bytes()_.
   * @sa ssize_t offset(int x, int y) const
   * @sa ssize_t offset(int x, int y, int z)  const
   * @sa ssize_t offset(int x,int y, int z, int w)  const
   * @sa ssize_t offset(const int *c)  const
   */
  inline ssize_t offset(int x) const
  {
    const Py_ssize_t* m = strides();
    assert((unsigned int)x<shape()[0]);
    return NUMPY_HPP_OFFSET1;
  }
  
  inline ssize_t offset(int x, int y) const
  {
    const Py_ssize_t* m = strides();
    assert((Py_ssize_t)x<shape()[0]);
    assert((Py_ssize_t)y<shape()[1]);
    return NUMPY_HPP_OFFSET2;
  }
  
  inline ssize_t offset(int x, int y, int z) const
  {
    const Py_ssize_t* m = strides();
    assert((Py_ssize_t)x<shape()[0]);
    assert((Py_ssize_t)y<shape()[1]);
    assert((Py_ssize_t)z<shape()[2]);
    return NUMPY_HPP_OFFSET3;
  }
  
  inline ssize_t offset(int x,int y, int z, int w) const
  {
    const Py_ssize_t* m = strides();
    assert((Py_ssize_t)x<shape()[0]);
    assert((Py_ssize_t)y<shape()[1]);
    assert((Py_ssize_t)z<shape()[2]);
    assert((Py_ssize_t)w<shape()[3]);
    return NUMPY_HPP_OFFSET4;
  }

  inline ssize_t offset(const int *c) const /// _c_ must point to an array of at least rank() items.
  {
    const Py_ssize_t* m = strides();
    int r = rank();
    Py_ssize_t q = 0;
    for(int i=0; i<r; ++i)
    {
      assert((Py_ssize_t)(c[i])<shape()[i]);
      q += c[i]*m[i];
    }
    return q;
  }

#undef NUMPY_HPP_OFFSET1
#undef NUMPY_HPP_OFFSET2
#undef NUMPY_HPP_OFFSET3
#undef NUMPY_HPP_OFFSET4
  
  /// Access to the array's memory block.
  char* bytes();
  const char* bytes() const { return const_cast<arraytbase*>(this)->bytes(); }
};

/**
@brief This class defines operators () and [] to allow for direct memory access to 
array elements of type T.
*/
template<class T>
class arrayt : public arraytbase
{
public:
  arrayt() : arraytbase() {}
  /**
   * @brief Construct a new instance from the base class.
   * 
   * The constructors check if the referenced array is behaved (in numpy terms),
   * which means that it must satisfy certain alignment and byte order criteria.
   * (See numpy documentation). The constructor also checks if sizeof(T) is equal
   * to the numpy data type of a. If the checks fail an exception of std::invalid_argument
   * is raised.
   * 
   * \sa arrayt(object const &a)
   */
  arrayt(arraytbase const &a) : arraytbase(a.getObject(), sizeof(T)) 
  {
  }
  
  /// Construct a new  instance from a boost::python object.
  arrayt(object const &a) : arraytbase(a, sizeof(T))
  {
  }

  /// Take hold of another array. Same rules as for the constructors apply.
  void init(object const &a_)
  {
    this->~arrayt<T>();
    new (this) arrayt<T>(a_);
  }

  T& operator()(int x)
  {
    return *((T*)(arraytbase::bytes()+offset(x)));
  }
  
  T& operator()(int x, int y)
  {
    return *((T*)(arraytbase::bytes()+offset(x,y)));
  }
  
  T& operator()(int x, int y, int z)
  {
    return *((T*)(arraytbase::bytes()+offset(x,y,z)));
  }
  
  T& operator()(int x, int y, int z, int w)
  {
    return *((T*)(arraytbase::bytes()+offset(x,y,z,w)));
  }
  
  T& operator()(int *c)
  {
    return *((T*)(arraytbase::bytes()+offset(c)));
  }
  
  T& operator[](int i)
  {
    return *((T*)(arraytbase::bytes()+offset(i)));
  }

  T* data() { return reinterpret_cast<T*>(bytes()); }
  
  
  T operator()(int x) const
  {
    return *((T*)(arraytbase::bytes()+offset(x)));
  }
  
  T operator()(int x, int y) const
  {
    return *((T*)(arraytbase::bytes()+offset(x,y)));
  }
  
  T operator()(int x, int y, int z) const
  {
    return *((T*)(arraytbase::bytes()+offset(x,y,z)));
  }
  
  T operator()(int x, int y, int z, int w) const
  {
    return *((T*)(arraytbase::bytes()+offset(x,y,z,w)));
  }
  
  T operator()(int *c) const
  {
    return *((T*)(arraytbase::bytes()+offset(c)));
  }
  
  T operator[](int i) const
  {
    return *((T*)(arraytbase::bytes()+offset(i)));
  }
  
  const T* data() const { return reinterpret_cast<T*>(bytes()); }
};


namespace mw_py_impl
{
  template<class T, class Idx1, class Idx2, class Idx3>
  inline void gridded_data_ccons(T* dst, const T* src, 
                                 const Idx1 *dims, 
                                 const Idx2 *dst_strides, 
                                 const Idx3 *src_strides, 
                                 boost::mpl::int_<0>)
  {
    for (int i=0; i<dims[0]; ++i)
    {
      new (dst) T(*src);
      dst += dst_strides[0];
      src += src_strides[0];
    }
  }
  
  template<class T, class Idx1, class Idx2, class Idx3, int dim>
  inline void gridded_data_ccons(T* dst, const T* src, 
                                 const Idx1* dims, 
                                 const Idx2 *dst_strides, 
                                 const Idx3 *src_strides, 
                                 boost::mpl::int_<dim>)
  {
    for (int i=0; i<dims[dim]; ++i)
    {
      gridded_data_ccons(dst, src, dims, dst_strides, src_strides, boost::mpl::int_<dim-1>());
      dst += dst_strides[dim];
      src += src_strides[dim];
    }
  }
}


/**
 * @brief Copy contents of n-dimensional non-contiguous arrays.
 * 
 * Strides given in *number of items*, *not in bytes*
 */
template<class T, int rank>
static array copy(const int* dims, const T* src, const int *strides)
{
  int item_type = getItemtype<T>();
  typename arrayt<T>::ssize_t arr_dims[MAX_DIM];
  for (int i=0; i<rank; ++i) arr_dims[i] = dims[i];
  arrayt<T> arr(empty(rank, arr_dims, item_type));

  typename arrayt<T>::ssize_t arr_strides[MAX_DIM];
  for (int i=0; i<rank; ++i) arr_strides[i] = arr.strides()[i]/arr.itemsize();
  
  T* dst = arr.data();
  mw_py_impl::gridded_data_ccons<T>(dst, src, dims, arr_strides, strides, boost::mpl::int_<rank-1>());
  
  return arr.getObject();
}

template<class T, int rank>
static void copy(T* dst, const int *dims, const int* strides, const array &pyarr)
{
  arrayt<T> arr(pyarr);

  typename arrayt<T>::ssize_t arr_strides[MAX_DIM];
  for (int i=0; i<rank; ++i) {
    assert(arr.shape()[i] == dims[i]);
    arr_strides[i] = arr.strides()[i]/arr.itemsize();
  }
  
  const T* src = arr.data();
  mw_py_impl::gridded_data_ccons<T>(dst, src, arr.shape(), strides, arr_strides, boost::mpl::int_<rank-1>());
}


} } }

#endif
