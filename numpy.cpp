#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION 
#include "numpy.hpp"
#include "numpy/ndarrayobject.h"
#include "assert.h"
#include <stdexcept>

namespace py = boost::python;

namespace boost { namespace python { namespace numeric {

namespace mw_py_impl
{

/* sometimes scalar numpy arrays end up as arguments to c++ function calls. 
 * For these cases we need automatic conversion functions such as these.
 * However only conversion to int, float and double is current implemented. 
 */
struct int_from_numpyint32
{
    static void Register()
    {
      boost::python::converter::registry::push_back(
        &convertible,
        &construct,
        boost::python::type_id<int>());
    }

    static void* convertible(PyObject* obj_ptr)
    {
      if (PyArray_IsScalar(obj_ptr, Int32) ||
          PyArray_IsScalar(obj_ptr, Int64) ||
          PyArray_IsScalar(obj_ptr, Int16) ||
          PyArray_IsScalar(obj_ptr, Int8) ||
          PyArray_IsScalar(obj_ptr, UInt8))
        return obj_ptr;
      else
        return 0;
    }

    static void construct(
      PyObject* obj_ptr,
      boost::python::converter::rvalue_from_python_stage1_data* data)
    {
      void* storage = ((boost::python::converter::rvalue_from_python_storage<int>*)data)->storage.bytes;
      data->convertible = storage;
      
      union
      {
        int int_data;
        short short_data;
        long long longlong_data;
        char char_data;
        unsigned char uchar_data;
        unsigned char buffer[128];
      } buffer;
      
      if (PyArray_IsScalar(obj_ptr, Int8))
      {
        PyArray_CastScalarToCtype(obj_ptr, buffer.buffer, PyArray_DescrFromType(NPY_INT8));
        new (storage) int(buffer.char_data);
      }
      else if (PyArray_IsScalar(obj_ptr, UInt8))
      {
        PyArray_CastScalarToCtype(obj_ptr, buffer.buffer, PyArray_DescrFromType(NPY_UINT8));
        new (storage) int(buffer.uchar_data);
      }
      else if (PyArray_IsScalar(obj_ptr, UInt16))
      {
        PyArray_CastScalarToCtype(obj_ptr, buffer.buffer, PyArray_DescrFromType(NPY_INT16));
        new (storage) int(buffer.short_data);
      }
      else if (PyArray_IsScalar(obj_ptr, Int32))
      {
        PyArray_CastScalarToCtype(obj_ptr, buffer.buffer, PyArray_DescrFromType(NPY_INT32));
        new (storage) int(buffer.int_data);
      }
      else if (PyArray_IsScalar(obj_ptr, Int64))
      {
        PyArray_CastScalarToCtype(obj_ptr, buffer.buffer, PyArray_DescrFromType(NPY_INT64));
        new (storage) int(buffer.longlong_data);
      }
      else
        throw std::invalid_argument("numpy int type conversion error");
    }
};

template<class T>
struct from_numpyfloat
{
    static void Register()
    {
      boost::python::converter::registry::push_back(
        &convertible,
        &construct,
        boost::python::type_id<T>());
    }

    static void* convertible(PyObject* obj_ptr)
    {
      if (!PyArray_IsScalar(obj_ptr, Float32) && !PyArray_IsScalar(obj_ptr, Float64)) return 0;
      return obj_ptr;
    }

    static void construct(
      PyObject* obj_ptr,
      boost::python::converter::rvalue_from_python_stage1_data* data)
    {
      void* storage = ((boost::python::converter::rvalue_from_python_storage<T>*)data)->storage.bytes;
      data->convertible = storage;

      union
      {
        float float_data;
        double double_data;
        unsigned char buffer[128];
      } buffer;
      
      if (PyArray_IsScalar(obj_ptr, Float32))
      {
        PyArray_CastScalarToCtype(obj_ptr, buffer.buffer, PyArray_DescrFromType(NPY_FLOAT32));
        new (storage) T(buffer.float_data);
      }
      else if (PyArray_IsScalar(obj_ptr, Float64))
      {
        PyArray_CastScalarToCtype(obj_ptr, buffer.buffer, PyArray_DescrFromType(NPY_FLOAT64));
        new (storage) T(buffer.double_data);
      }
      else
        throw std::invalid_argument("numpy int type conversion error");
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
  mw_py_impl::int_from_numpyint32::Register();
  mw_py_impl::from_numpyfloat<float>::Register();
  mw_py_impl::from_numpyfloat<double>::Register();
  
  py::numeric::array::set_module_and_type("numpy", "ndarray"); // use numpy
}


array zeros(int rank, const int *dims, int type)
{
  npy_intp tmp[NPY_MAXDIMS];
  for(int i=0; i<rank; ++i) tmp[i] = dims[i];
  PyObject* p = PyArray_ZEROS(rank,tmp,type,true);
  return array(handle<>(p));
}


array empty(int rank, const int *dims, int type )
{
  npy_intp tmp[NPY_MAXDIMS];
  for(int i=0; i<rank; ++i) tmp[i] = dims[i];
  PyObject* p = PyArray_EMPTY(rank, tmp, type, true);
  return array(handle<>(p));
}



int getItemtype(const array &a)
{ 
  const PyArrayObject* p = getPyArrayObjectPtr(a);
  int t = PyArray_TYPE(p);
  return t;
}


arraytbase::arraytbase() :
  obj(object())
{

}

arraytbase::arraytbase(const object& a_) :
  obj(object()),
  bytes_(NULL),
  r(0)
{
  memset(m,0,sizeof(int)*MAX_DIM);
  memset(s,0,sizeof(int)*MAX_DIM);

  if (a_.is_none()) return;

  array a = obj = extract<array>(a_);
  PyArrayObject* p = getPyArrayObjectPtr(a);
  
  bool is_behaved = PyArray_ISBEHAVED(p);
  if (!is_behaved)
    throw std::invalid_argument("arrayt: numpy array is not behaved");
  
  // directly from the python object
  //int t = getItemtype(a);
  r = PyArray_NDIM(p);
  assert(r < MAX_DIM);
  itemtype_ = PyArray_TYPE(p);
  itemsize_ = PyArray_ITEMSIZE(p);
  bytes_ = (unsigned char*)PyArray_BYTES(p);
  // copy stuff
  Py_ssize_t* p_strides = PyArray_STRIDES(p);
  Py_ssize_t* p_dims = PyArray_DIMS(p);
  int i=0;
  for(; i<r; ++i)
  {
    m[i] = p_strides[i];
    s[i] = p_dims[i];
  }
  for(; i<MAX_DIM; ++i)
    s[i] = 1;
}


bool arraytbase::isCContiguous() const
{
  const PyArrayObject* p = getPyArrayObjectPtr(obj);
  return PyArray_IS_C_CONTIGUOUS(p);
}


bool arraytbase::isWriteable() const
{
  const PyArrayObject* p = getPyArrayObjectPtr(obj);
  return PyArray_ISWRITEABLE(p);
}


bool arraytbase::isFContiguous() const
{
  const PyArrayObject* p = getPyArrayObjectPtr(obj);
  return PyArray_IS_F_CONTIGUOUS(p);
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