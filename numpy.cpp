#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION 
#include "numpy.hpp"
#include "numpy/ndarrayobject.h"
#include "assert.h"
#include <stdexcept>
#include <cstring>

namespace py = boost::python;
namespace np = boost::python::numeric;

namespace boost { namespace python { namespace numeric {

namespace mw_py_impl
{

#if __cplusplus > 200000L
static_assert(np::MAX_DIM == NPY_MAXDIMS, "MAX_DIMS should be equal to NPY_MAXDIMS");
#endif

/* Sometimes scalar numpy arrays end up as arguments to c++ function calls. 
 * For these cases we need automatic conversion functions such as the following.
 */
template<class T>
struct from_numpy_scalar
{
    static PyArray_Descr* descr;
  
    static void Register()
    {
      int itemtype = np::getItemtype<T>();
      descr    = PyArray_DescrFromType(itemtype);
      
      boost::python::converter::registry::push_back(
        &convertible,
        &construct,
        boost::python::type_id<T>());
    }

    static void* convertible(PyObject* obj_ptr)
    {
      if (!PyArray_CheckScalar(obj_ptr)) return NULL;
      PyArray_Descr* objdescr = PyArray_DescrFromScalar(obj_ptr);
      bool ok = PyArray_CanCastTypeTo(objdescr, descr, NPY_SAME_KIND_CASTING);
      Py_DECREF(objdescr);
      return ok ? obj_ptr : NULL;
    }

    static void construct(
      PyObject* obj_ptr,
      boost::python::converter::rvalue_from_python_stage1_data* data)
    {
      void* storage = ((boost::python::converter::rvalue_from_python_storage<T>*)data)->storage.bytes;
      data->convertible = storage;
      
      union
      {
        T data;
        unsigned char buffer[sizeof(T)];
      } buffer;
      memset(buffer.buffer, 0, sizeof(T));
      
      PyArray_CastScalarToCtype(obj_ptr, buffer.buffer, descr);
      new (storage) T(buffer.data);
    }
};

template<class T>
PyArray_Descr* from_numpy_scalar<T>::descr = NULL;



template<class T>
struct to_numpyarrayt
{
  typedef np::arrayt<T> ArrayType;
  
  static void Register()
  {
    boost::python::converter::registry::push_back(
      &convertible,
      &construct,
      boost::python::type_id<ArrayType>());
  }
  
  static void* convertible(PyObject* obj_ptr)
  {
    bool ok = PyArray_Check(obj_ptr);
    ok &= np::isCompatibleType<T>(PyArray_TYPE((PyArrayObject*)obj_ptr));
    return ok ? obj_ptr : 0;
  }
  
  static void construct(
    PyObject* obj_ptr,
    boost::python::converter::rvalue_from_python_stage1_data* data)
  {
    void* storage = ((boost::python::converter::rvalue_from_python_storage<ArrayType>*)data)->storage.bytes;
    data->convertible = storage;
    
    new (storage) arrayt<T>(py::object(py::borrowed(obj_ptr)));
  }
};


struct to_toarraytbase
{
  typedef np::arraytbase ArrayType;
  
  static void Register()
  {
    boost::python::converter::registry::push_back(
      &convertible,
      &construct,
      boost::python::type_id<ArrayType>());
  }
  
  static void* convertible(PyObject* obj_ptr)
  {
    bool ok = PyArray_Check(obj_ptr);
    return ok ? obj_ptr : 0;
  }
  
  static void construct(
    PyObject* obj_ptr,
    boost::python::converter::rvalue_from_python_stage1_data* data)
  {
    void* storage = ((boost::python::converter::rvalue_from_python_storage<ArrayType>*)data)->storage.bytes;
    data->convertible = storage;
    new (storage) arraytbase(py::object(py::borrowed(obj_ptr)));
  }
};


PyArrayObject* getPyArrayObjectPtr(py::object &o)
{
  return reinterpret_cast<PyArrayObject*>(o.ptr());
}

const PyArrayObject* getPyArrayObjectPtr(const py::object &o)
{
  return reinterpret_cast<const PyArrayObject*>(o.ptr());
}

} // impl namespace



using mw_py_impl::getPyArrayObjectPtr;

void importNumpyAndRegisterTypes()
{
  import_array1(); // this is from the numpy c-API and import the numpy module into python
  
  // these are boost python type conversions
  mw_py_impl::from_numpy_scalar<bool>::Register();
  mw_py_impl::from_numpy_scalar<char>::Register();
  mw_py_impl::from_numpy_scalar<short>::Register();
  mw_py_impl::from_numpy_scalar<int>::Register();
  mw_py_impl::from_numpy_scalar<long>::Register();
  mw_py_impl::from_numpy_scalar<long long>::Register();

  mw_py_impl::from_numpy_scalar<unsigned char>::Register();
  mw_py_impl::from_numpy_scalar<unsigned short>::Register();
  mw_py_impl::from_numpy_scalar<unsigned int>::Register();
  mw_py_impl::from_numpy_scalar<unsigned long>::Register();
  mw_py_impl::from_numpy_scalar<unsigned long long>::Register();
  
  mw_py_impl::from_numpy_scalar<float>::Register();
  mw_py_impl::from_numpy_scalar<double>::Register();
  
  mw_py_impl::to_toarraytbase::Register();
  mw_py_impl::to_numpyarrayt<bool>::Register();
  mw_py_impl::to_numpyarrayt<char>::Register();
  mw_py_impl::to_numpyarrayt<short>::Register();
  mw_py_impl::to_numpyarrayt<int>::Register();
  mw_py_impl::to_numpyarrayt<long>::Register();
  mw_py_impl::to_numpyarrayt<long long>::Register();
  mw_py_impl::to_numpyarrayt<float>::Register();
  mw_py_impl::to_numpyarrayt<double>::Register();

  mw_py_impl::to_numpyarrayt<unsigned char>::Register();
  mw_py_impl::to_numpyarrayt<unsigned short>::Register();
  mw_py_impl::to_numpyarrayt<unsigned int>::Register();
  mw_py_impl::to_numpyarrayt<unsigned long>::Register();
  mw_py_impl::to_numpyarrayt<unsigned long long>::Register();
  
  py::numeric::array::set_module_and_type("numpy", "ndarray"); // use numpy
}


array zeros(int rank, const Py_ssize_t *dims, int type)
{
  npy_intp* tmp = const_cast<npy_intp*>(dims);
  PyObject* p = PyArray_ZEROS(rank,tmp,type,true);
  return array(handle<>(p));
}


array empty(int rank, const Py_ssize_t *dims, int type )
{
  npy_intp* tmp = const_cast<npy_intp*>(dims);
  PyObject* p = PyArray_EMPTY(rank, tmp, type, true);
  return array(handle<>(p));
}



int getItemtype(const array &a)
{ 
  const PyArrayObject* p = getPyArrayObjectPtr(a);
  int t = PyArray_TYPE(p);
  return t;
}


arraytbase::arraytbase(const object& a_) :
  obj(object()),
  objptr(NULL)
{
  construct(a_, -1);
}

arraytbase::arraytbase(const object& a_, int typesize) :
  obj(object()),
  objptr(NULL)
{
  construct(a_, typesize);
}


void arraytbase::construct(object const &a_, int typesize)
{ 
  obj = extract<array>(a_);
  if (obj.is_none()) return;
  
  objptr = getPyArrayObjectPtr(obj);
  
  bool is_behaved = PyArray_ISBEHAVED(objptr);
  if (!is_behaved)
    throw std::invalid_argument("arrayt: numpy array is not behaved");

  if (typesize>0 && typesize != itemsize())
    throw std::invalid_argument("arrayt: array itemsize does not match template argument"); 
}


int arraytbase::itemtype() const
{
  return PyArray_TYPE(objptr);
}

int arraytbase::itemsize() const
{
  return PyArray_ITEMSIZE(objptr);
}

int arraytbase::rank() const
{
  return PyArray_NDIM(objptr);
}

const Py_ssize_t* arraytbase::shape() const
{
  return PyArray_DIMS(objptr);
}


const Py_ssize_t* arraytbase::strides() const
{
  return PyArray_STRIDES(objptr);
}


bool arraytbase::isCContiguous() const
{
  return PyArray_IS_C_CONTIGUOUS(objptr);
}


bool arraytbase::isWriteable() const
{
  return PyArray_ISWRITEABLE(objptr);
}


bool arraytbase::isFContiguous() const
{
  return PyArray_IS_F_CONTIGUOUS(objptr);
}


char* arraytbase::bytes()
{
  return PyArray_BYTES(objptr);
}


template<class T>
bool isCompatibleType(int id)
{
  return false;
}

#define DEF_TYPE_COMPATIBILITY1(T, npyT1) \
  template<> bool isCompatibleType<T>(int id)  {  return id == npyT1; }
#define DEF_TYPE_COMPATIBILITY2(T, npyT1, npyT2) \
  template<> bool isCompatibleType<T>(int id)  {  return id == npyT1 || id == npyT2; }
#define DEF_TYPE_COMPATIBILITY3(T, npyT1, npyT2, npyT3) \
  template<> bool isCompatibleType<T>(int id)  {  return id == npyT1 || id == npyT2 || id == npyT3; }  
#define DEF_TYPE_COMPATIBILITY4(T, npyT1, npyT2, npyT3, npyT4) \
  template<> bool isCompatibleType<T>(int id)  {  return id == npyT1 || id == npyT2 || id == npyT3 || id == npyT4; }  
  
  
DEF_TYPE_COMPATIBILITY1(PyObject*, NPY_OBJECT)
DEF_TYPE_COMPATIBILITY1(bool, NPY_BOOL)
DEF_TYPE_COMPATIBILITY3(char, NPY_INT8, NPY_BYTE, NPY_BOOL)
DEF_TYPE_COMPATIBILITY3(unsigned char, NPY_UINT8, NPY_UBYTE, NPY_BOOL)
DEF_TYPE_COMPATIBILITY2(short, NPY_INT16, NPY_SHORT)
DEF_TYPE_COMPATIBILITY2(unsigned short, NPY_UINT16, NPY_USHORT)
DEF_TYPE_COMPATIBILITY2(int, NPY_INT32, NPY_INT)  // NPY_INT is always 32 bit
DEF_TYPE_COMPATIBILITY2(unsigned int, NPY_UINT32, NPY_UINT)
// different sizes of type long, see here http://en.cppreference.com/w/cpp/language/types
#if NPY_BITSOF_LONG == 64 // defined in npy_common.h
  DEF_TYPE_COMPATIBILITY3(unsigned long, NPY_UINT64, NPY_ULONG, NPY_ULONGLONG)
  DEF_TYPE_COMPATIBILITY3(long, NPY_INT64, NPY_LONG, NPY_LONGLONG) // NPY_LONG can be NPY_INT or NPY_LONGLONG; NPY_LONGLONG and NPY_INT64 are always 64 bit
#elif NPY_BITSOF_LONG == 32
  DEF_TYPE_COMPATIBILITY2(unsigned long, NPY_UINT32, NPY_UINT)
  DEF_TYPE_COMPATIBILITY2(long, NPY_INT32, NPY_INT)
#endif
DEF_TYPE_COMPATIBILITY3(long long, NPY_INT64, NPY_LONG, NPY_LONGLONG)
DEF_TYPE_COMPATIBILITY3(unsigned long long, NPY_UINT64, NPY_ULONG, NPY_ULONGLONG)
DEF_TYPE_COMPATIBILITY1(float, NPY_FLOAT32)
DEF_TYPE_COMPATIBILITY2(double, NPY_FLOAT64, NPY_DOUBLE)

#define DEF_ARRY_TYPE2ID(t,id)\
template<> int getItemtype<t>() {\
return id; \
};

DEF_ARRY_TYPE2ID(PyObject*, NPY_OBJECT);
DEF_ARRY_TYPE2ID(float,NPY_FLOAT32);
DEF_ARRY_TYPE2ID(double,NPY_FLOAT64);
DEF_ARRY_TYPE2ID(int,NPY_INT);
DEF_ARRY_TYPE2ID(long,NPY_LONG);
DEF_ARRY_TYPE2ID(long long ,NPY_LONGLONG);
DEF_ARRY_TYPE2ID(short,NPY_SHORT);
DEF_ARRY_TYPE2ID(char,NPY_BYTE);
DEF_ARRY_TYPE2ID(bool,NPY_BOOL); // lets hope that bool is one byte long
DEF_ARRY_TYPE2ID(unsigned int,NPY_UINT);
DEF_ARRY_TYPE2ID(unsigned long,NPY_ULONG);
DEF_ARRY_TYPE2ID(unsigned short,NPY_USHORT);
DEF_ARRY_TYPE2ID(unsigned char,NPY_UBYTE);
DEF_ARRY_TYPE2ID(unsigned long long ,NPY_ULONGLONG);


} } } // namespace