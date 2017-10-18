/*
 C *opyright (c) 2016 Michael Welter
 
 This file is part of numpycpp.
 
 numpycpp is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 
 numpycpp is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with numpycpp.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <iostream> 
#include <sstream>
#include <boost/python.hpp>
#include <boost/python/object.hpp>
#if BOOST_VERSION>106300
  #include <boost/python/numpy.hpp>
  #include <boost/python/numpy/ndarray.hpp>
  #include "/usr/lib/python2.7/site-packages/numpy/core/include/numpy/ndarrayobject.h"
  namespace np = boost::python::numpy;
#else
  #include "numpy.hpp"
  namespace nm = boost::python::numeric;
#endif

namespace py = boost::python;


using namespace std;

// #define CAST_TO_PPYARRAYOBJECT(p) ((PyArrayObject*)(p))
// template<class T>
// bool isCompatibleType(int id)
// {
//   return false;
// }
// 
// /** @cond */
// #define DEF_TYPE_COMPATIBILITY1(T, npyT1) \
//   template<> bool isCompatibleType<T>(int id)  {  return id == npyT1; }
// #define DEF_TYPE_COMPATIBILITY2(T, npyT1, npyT2) \
//   template<> bool isCompatibleType<T>(int id)  {  return id == npyT1 || id == npyT2; }
// #define DEF_TYPE_COMPATIBILITY3(T, npyT1, npyT2, npyT3) \
//   template<> bool isCompatibleType<T>(int id)  {  return id == npyT1 || id == npyT2 || id == npyT3; }  
// #define DEF_TYPE_COMPATIBILITY4(T, npyT1, npyT2, npyT3, npyT4) \
//   template<> bool isCompatibleType<T>(int id)  {  return id == npyT1 || id == npyT2 || id == npyT3 || id == npyT4; }  
//   
//   
// DEF_TYPE_COMPATIBILITY1(PyObject*, NPY_OBJECT)
// DEF_TYPE_COMPATIBILITY1(bool, NPY_BOOL)
// DEF_TYPE_COMPATIBILITY3(char, NPY_INT8, NPY_BYTE, NPY_BOOL)
// DEF_TYPE_COMPATIBILITY3(unsigned char, NPY_UINT8, NPY_UBYTE, NPY_BOOL)
// DEF_TYPE_COMPATIBILITY2(short, NPY_INT16, NPY_SHORT)
// DEF_TYPE_COMPATIBILITY2(unsigned short, NPY_UINT16, NPY_USHORT)
// DEF_TYPE_COMPATIBILITY2(int, NPY_INT32, NPY_INT)  // NPY_INT is always 32 bit
// DEF_TYPE_COMPATIBILITY2(unsigned int, NPY_UINT32, NPY_UINT)
// // different sizes of type long, see here http://en.cppreference.com/w/cpp/language/types
// #if NPY_BITSOF_LONG == 64 // defined in npy_common.h
//   DEF_TYPE_COMPATIBILITY3(unsigned long, NPY_UINT64, NPY_ULONG, NPY_ULONGLONG)
//   DEF_TYPE_COMPATIBILITY3(long, NPY_INT64, NPY_LONG, NPY_LONGLONG) // NPY_LONG can be NPY_INT or NPY_LONGLONG; NPY_LONGLONG and NPY_INT64 are always 64 bit
// #elif NPY_BITSOF_LONG == 32
//   DEF_TYPE_COMPATIBILITY2(unsigned long, NPY_UINT32, NPY_UINT)
//   DEF_TYPE_COMPATIBILITY2(long, NPY_INT32, NPY_INT)
// #endif
// DEF_TYPE_COMPATIBILITY3(long long, NPY_INT64, NPY_LONG, NPY_LONGLONG)
// DEF_TYPE_COMPATIBILITY3(unsigned long long, NPY_UINT64, NPY_ULONG, NPY_ULONGLONG)
// DEF_TYPE_COMPATIBILITY1(float, NPY_FLOAT32)
// DEF_TYPE_COMPATIBILITY2(double, NPY_FLOAT64, NPY_DOUBLE)

/** @brief Equal to NPY_MAX_DIMS. Provided so that the numpy headers don't have to be included.
*/
enum {
  MAX_DIM = 32
};

template<class T>
void print(const T &x, std::ostream &os)
{
  os << x;
}

template<>
void print<char>(const char &x, std::ostream &os)
{
  os << (int)x;
}

template<>
void print<PyObject*>(PyObject* const &x, std::ostream &os)
{
  // increase reference count, so the object is not destructed when p goes out of scope. 
  // See http://www.boost.org/doc/libs/1_53_0/libs/python/doc/tutorial/doc/html/python/object.html#python.creating_python_object
  py::object p(py::borrowed(x));
  std::string s = py::extract<std::string>(py::str(p));
  cout << s;
}
#if BOOST_VERSION>106300
#else
int getItemtype(const py::object &a)
{ 
  if (PyArray_Check(a.ptr()))
  {
    const PyArrayObject* p = CAST_TO_PPYARRAYOBJECT(a.ptr());
    int t = PyArray_TYPE(p);
    return t;
  }
  else
    return -1;
}

#define DEF_ARRY_TYPE2ID(t,id)\
template<> int getItemtype<t>() {\
return id; \
};

#endif 

#if BOOST_VERSION>106300
void printElements(const np::ndarray &a, int dim, int *indices)
{
  if (dim >= a.get_nd())
  {
    cout << "(";
    int i = 0;
    while (i<a.get_nd()-1)  cout << indices[i++] << ',';
    cout << indices[i] << ") = ";
    print(py::extract<int>(a[indices]), cout); 
    cout << endl;
  }
  else
  {
    const int n = a.get_shape()[dim];
    for (indices[dim]=0; indices[dim]<n; ++indices[dim])
    {
      printElements(a, dim+1, indices);
    }
  }
}
#else
template<class T>
void printElements(const np::arrayt<T> &a, int dim, int *indices)
{
  if (dim >= a.rank())
  {
    cout << "(";
    int i = 0;
    while (i<a.rank()-1)  cout << indices[i++] << ',';
    cout << indices[i] << ") = ";
    print(a(indices), cout); 
    cout << endl;
  }
  else
  {
    const int n = a.shape()[dim];
    for (indices[dim]=0; indices[dim]<n; ++indices[dim])
    {
      printElements(a, dim+1, indices);
    }
  }
}
#endif

#if BOOST_VERSION>106300
template<class T>
void handleType(np::ndarray arr, bool printContents)
{
  if (np::equivalent(np::dtype::get_builtin<T>(), arr.get_dtype()))
  {
    cout << "data of type: <" << typeid(T).name() << "> will be handeled" << endl;
  }
//   if (isCompatibleType<T>(arr.get_dtype()))
//   {
    if (printContents)
    {
      cout << "Printing array of type " << typeid(T).name() << endl;
      int indices[MAX_DIM];
      printElements(np::ndarray(arr), 0, indices);
    }
    else
    {
      cout << "mapped c++ type = " << typeid(T).name() << endl;
    }
//   }    
}
#else
template<class T>
void handleType(np::arraytbase arr, bool printContents)
{
  if (np::getItemtype<T>() == arr.itemtype())
  {
    cout << "getItemtype<" << typeid(T).name() << "> matches type of array" << endl;
  }
  if (np::isCompatibleType<T>(arr.itemtype()))
  {
    if (printContents)
    {
      cout << "Printing array of type " << typeid(T).name() << endl;
      int indices[np::MAX_DIM];
      printElements(np::arrayt<T>(arr), 0, indices);
    }
    else
    {
      cout << "mapped c++ type = " << typeid(T).name() << endl;
    }
  }    
}
#endif

#if BOOST_VERSION>106300
void printArrayInfo(const np::ndarray &arr)
{
  int i;
  cout << "get_shape()          = " << arr.get_shape() << endl;
  //cout << "get_dtype()          = " << arr.get_dtype() << endl;
  //cout << "itemsize()      = " << arr.itemsize() << endl;
  //cout << "isWriteable()   = " << arr.isWriteable() << endl;
  //cout << "isCContiguous() = " << arr.isCContiguous() << endl;
  //cout << "isFContiguous() = " << arr.isFContiguous() << endl;
  cout << "shape()         = ";
  for (i=0; i<arr.get_nd()-1; ++i)
    cout << arr.shape(i) << ',';
  cout << arr.shape(i) << endl;
  cout << "strides()       = ";
  for (i=0; i<arr.get_nd()-1; ++i)
    cout << arr.strides(i) << ',';
  cout << arr.strides(i) << endl;
}
#else
void printArrayInfo(const np::arraytbase &arr)
{
  int i;
  cout << "rank()          = " << arr.rank() << endl;
  cout << "itemtype()      = " << arr.itemtype() << endl;
  cout << "itemsize()      = " << arr.itemsize() << endl;
  cout << "isWriteable()   = " << arr.isWriteable() << endl;
  cout << "isCContiguous() = " << arr.isCContiguous() << endl;
  cout << "isFContiguous() = " << arr.isFContiguous() << endl;
  cout << "shape()         = ";
  for (i=0; i<arr.rank()-1; ++i)
    cout << arr.shape()[i] << ',';
  cout << arr.shape()[i] << endl;
  cout << "strides()       = ";
  for (i=0; i<arr.rank()-1; ++i)
    cout << arr.strides()[i] << ',';
  cout << arr.strides()[i] << endl;
}
#endif

#if BOOST_VERSION>106300
void printArray(np::ndarray pyarr, bool printContents)
{
  np::ndarray arr(pyarr);
  printArrayInfo(arr);
  
//   handleType<float>(arr, printContents);
//   handleType<PyObject*>(arr, printContents);
  
//   handleType<double>(arr, printContents);
//   
//   handleType<bool>(arr, printContents);
//   handleType<int>(arr, printContents);
//   handleType<short>(arr, printContents);
//   handleType<char>(arr, printContents);
//   handleType<long>(arr, printContents);
//   handleType<long long>(arr, printContents);
//   
//   handleType<unsigned int>(arr, printContents);
//   handleType<unsigned short>(arr, printContents);
//   handleType<unsigned char>(arr, printContents);
//   handleType<unsigned long>(arr, printContents);
//   handleType<unsigned long long>(arr, printContents);
}
#else
void printArray(nm::array pyarr, bool printContents)
{
  np::arraytbase arr(pyarr);
  printArrayInfo(arr);
  
  handleType<PyObject*>(arr, printContents);
  handleType<float>(arr, printContents);
  handleType<double>(arr, printContents);
  
  handleType<bool>(arr, printContents);
  handleType<int>(arr, printContents);
  handleType<short>(arr, printContents);
  handleType<char>(arr, printContents);
  handleType<long>(arr, printContents);
  handleType<long long>(arr, printContents);
  
  handleType<unsigned int>(arr, printContents);
  handleType<unsigned short>(arr, printContents);
  handleType<unsigned char>(arr, printContents);
  handleType<unsigned long>(arr, printContents);
  handleType<unsigned long long>(arr, printContents);
}
#endif

#if BOOST_VERSION>106300
void printConvertedArray(np::ndarray arr)
{
  printArrayInfo(arr);
}
#else
template<class T>
void printConvertedArray(np::arrayt<T> arr)
{
  printArrayInfo(arr);
  handleType<T>(arr, true);
}
#endif

template<class T>
std::string scalar_to_str(const T &x)
{
  std::ostringstream os;
  os << x;
//   os << py::extract<T>(x);
//   os << py::extract<T>(x);
//   return std::string(py::extract<T>(x));
  return os.str();
}

template<>
std::string scalar_to_str(const char &x)
{
  std::ostringstream os;
  os << (int)x;
  return os.str();
}

#if BOOST_VERSION>106300
np::ndarray ReturnedFromCPP()
{
  np::ndarray ret = np::zeros(py::make_tuple(5), np::dtype::get_builtin<float>());
  for( int i=0; i<5; ++i)
    ret[i] = i;
  return ret;
}
#else
np::arraytbase ReturnedFromCPP()
{
  np::ssize_t dims = 5;
  np::arrayt<float> ret(np::empty(1, &dims, np::getItemtype<float>()));
  for (int i=0; i<dims; ++i)
    ret[i] = i;
  return ret;
}
#endif

#if BOOST_VERSION>106300

#else
py::object SumArrayT(np::arrayt<float> arr1, np::arrayt<float>  arr2)
{
  np::arrayt<float> ret(np::empty(arr1.rank(), arr1.dims(), arr1.itemtype()));
  
  const np::ssize_t *dims = arr1.dims();
  for (int i=0; i<dims[0]; ++i)
  {
    ret(i) = arr1(i) + arr2(i);
  }
  return ret.getObject();
}
#endif

#if BOOST_VERSION>106300
py::object SumNumericArray(np::ndarray arr1, np::ndarray arr2)
{
  //nm::array ret = py::extract<nm::array>(np::empty(1, &len, np::getItemtype(arr1)));
  //const Py_ssize_t len = py::extract<int>(py::getattr(arr1, "shape")[0]);
  //const Py_ssize_t len = arr1.get_nd();
  py::tuple shape = py::make_tuple(1,10);
  //np::dtype dtype = np::dtype::get_builtin<float>();
  np::ndarray ret = np::empty(shape,arr1.get_dtype());
  
  for (int i=0; i<arr1.get_nd(); ++i)
  {
    float x1 = py::extract<float>(arr1[i]);
    float x2 = py::extract<float>(arr2[i]);
    ret[i] = py::object(x1 + x2);
  }
  return ret;
}
#else
py::object SumNumericArray(nm::array arr1, nm::array arr2)
{
  const np::ssize_t len = py::extract<int>(py::getattr(arr1, "shape")[0]);
  nm::array ret = py::extract<nm::array>(np::empty(1, &len, np::getItemtype(arr1)));
  //py::tuple shape = py::make_tuple(1,len);
  //np::dtype dtype = np::dtype::get_builtin<float>();
  //np::ndarray ret = np::empty(shape,dtype);
  for (int i=0; i<len; ++i)
  {
    float x1 = py::extract<float>(arr1[i]);
    float x2 = py::extract<float>(arr2[i]);
    ret[i] = py::object(x1 + x2);
  }
  return ret;
}
#endif


BOOST_PYTHON_MODULE(libdemo)
{
  Py_Initialize();
#if BOOST_VERSION>106300
  cout<<"new boost numpy support detected"<<endl;
  np::initialize();
  py::def("printConvertedArray_int", printConvertedArray);
#else
  cout<<"NO boost numpy support detected"<<endl;
  np::importNumpyAndRegisterTypes();
  py::def("printConvertedArray_int", printConvertedArray<int>);
  py::def("SumArrayT", SumArrayT);
#endif
  
  py::def("printArray", printArray);
  
  py::def("int_to_str", scalar_to_str<int>);
  py::def("char_to_str", scalar_to_str<char>);
  py::def("float_to_str", scalar_to_str<float>);
  py::def("double_to_str", scalar_to_str<double>);
  
  py::def("SumNumericArray", SumNumericArray);
  py::def("ReturnedFromCPP", ReturnedFromCPP);
}


