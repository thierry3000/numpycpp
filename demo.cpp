#include <iostream> 
#include <boost/python/object.hpp>
//#include <boost/python/tuple.hpp>
//#include <boost/python/str.hpp>
//#include <boost/python/exec.hpp>
#include "numpy.hpp"

namespace py = boost::python;
namespace np = boost::python::numeric;
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


void printArray(np::array pyarr, bool printContents)
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
void print_stdout(const T &x)
{
  cout << x;
}


BOOST_PYTHON_MODULE(libdemo)
{
  np::importNumpyAndRegisterTypes();
  py::def("printArray", printArray);
  py::def("printConvertedArray_int", printConvertedArray<int>);
  py::def("print_int", print_stdout<int>);
  py::def("print_float", print_stdout<float>);
  py::def("print_double", print_stdout<double>);
}


