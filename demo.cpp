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
#include <boost/python/object.hpp>
#include "numpy.hpp"

namespace py = boost::python;
namespace np = boost::python::numpy;
namespace nm = boost::python::numeric;
using namespace std;

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


template<class T>
void printConvertedArray(np::arrayt<T> arr)
{
  printArrayInfo(arr);
  handleType<T>(arr, true);
}

template<class T>
std::string scalar_to_str(const T &x)
{
  std::ostringstream os;
  os << x;
  return os.str();
}

template<>
std::string scalar_to_str(const char &x)
{
  std::ostringstream os;
  os << (int)x;
  return os.str();
}


np::arraytbase ReturnedFromCPP()
{
  np::ssize_t dims = 5;
  np::arrayt<float> ret(np::empty(1, &dims, np::getItemtype<float>()));
  for (int i=0; i<dims; ++i)
    ret[i] = i;
  return ret;
}


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

py::object SumNumericArray(nm::array arr1, nm::array arr2)
{
  const np::ssize_t len = py::extract<int>(py::getattr(arr1, "shape")[0]);
  nm::array ret = py::extract<nm::array>(np::empty(1, &len, np::getItemtype(arr1)));
  for (int i=0; i<len; ++i)
  {
    float x1 = py::extract<float>(arr1[i]);
    float x2 = py::extract<float>(arr2[i]);
    ret[i] = py::object(x1 + x2);
  }
  return ret;
}



BOOST_PYTHON_MODULE(libdemo)
{
  np::importNumpyAndRegisterTypes();
  py::def("printArray", printArray);
  py::def("printConvertedArray_int", printConvertedArray<int>);
  py::def("char_to_str", scalar_to_str<char>);
  py::def("int_to_str", scalar_to_str<int>);
  py::def("float_to_str", scalar_to_str<float>);
  py::def("double_to_str", scalar_to_str<double>);
  py::def("SumArrayT", SumArrayT);
  py::def("SumNumericArray", SumNumericArray);
  py::def("ReturnedFromCPP", ReturnedFromCPP);
}


