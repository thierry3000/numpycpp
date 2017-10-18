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

#include <boost/python/numpy.hpp>
#include <boost/python/numpy/ndarray.hpp>


namespace np = boost::python::numpy;
namespace py = boost::python;


using namespace std;



/** @brief Equal to NPY_MAX_DIMS. Provided so that the numpy headers don't have to be included.
*/
enum {
  MAX_DIM = 32
};

template<class T>
void print(const T &x, std::ostream &os)
{
//   cout<< "in templated print" << endl;
//   cout.flush();
  std::string s = py::extract<std::string>(py::str(x));
  os << s;
//   os << x;
}

template<>
void print<char>(const char &x, std::ostream &os)
{
//   cout<< "in char print" << endl;
//   cout.flush();
  os << (int)x;
}

template<>
void print<PyObject*>(PyObject* const &x, std::ostream &os)
{
//   cout<< "in py print" << endl;
//   cout.flush();
  // increase reference count, so the object is not destructed when p goes out of scope. 
  // See http://www.boost.org/doc/libs/1_53_0/libs/python/doc/tutorial/doc/html/python/object.html#python.creating_python_object
  py::object p(py::borrowed(x));
  std::string s = py::extract<std::string>(py::str(p));
  cout << s;
}

template<class T>
void printElements(const np::ndarray &a, int dim, int *indices)
{
//   cout << "in printElements " << endl;
//   cout << "dim: " << dim << ", a.get_nd() " << a.get_nd() << endl;
  
  //T.F. unfortunatelly I am not able to access singel elements???
  std::cout << "Original array:\n" << py::extract<char const *>(py::str(a)) << std::endl;
//   if (dim > a.get_nd())
//   {
//     cout << "(";
//     int i = 0;
//     while (i<a.get_nd()-1)  cout << indices[i++] << ',';
//     cout << indices[i] << ") = ";
//     //note: this breaks!!!
//     int read = py::extract<T>(a[0]);
//     cout << endl;
//   }
//   else
//   {
//     const int n = a.shape(dim);
//     for (indices[dim]=0; indices[dim]<n; ++indices[dim])
//     {
//       printElements<T>(a, dim+1, indices);
//     }
//   }
}



template<class T>
void handleType(np::ndarray &arr, bool printContents)
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
      printElements<T>(arr, 0, indices);
    }
    else
    {
      cout << "mapped c++ type = " << typeid(T).name() << endl;
    }
//   }    
}


void printArrayInfo(const np::ndarray &arr)
{
  int i;
  cout << "we are in printArrayInfo" << endl;
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



void printArray(np::ndarray &pyarr, bool printContents)
{
  
  printArrayInfo(pyarr);
  
  handleType<float>(pyarr, printContents);
  handleType<PyObject*>(pyarr, printContents);
  
  handleType<double>(pyarr, printContents);
  
  handleType<bool>(pyarr, printContents);
  handleType<int>(pyarr, printContents);
  handleType<short>(pyarr, printContents);
  handleType<char>(pyarr, printContents);
  handleType<long>(pyarr, printContents);
  handleType<long long>(pyarr, printContents);
  
  handleType<unsigned int>(pyarr, printContents);
  handleType<unsigned short>(pyarr, printContents);
  handleType<unsigned char>(pyarr, printContents);
  handleType<unsigned long>(pyarr, printContents);
  handleType<unsigned long long>(pyarr, printContents);
}



template<class T>
void printConvertedArray(np::ndarray &arr)
{
  printArrayInfo(arr);
  cout << " going int handleType " <<endl;
  handleType<T>(arr, true);
}


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

np::ndarray ReturnedFromCPP()
{
  np::ndarray ret = np::zeros(py::make_tuple(5), np::dtype::get_builtin<float>());
  for( int i=0; i<5; ++i)
    ret[i] = i;
  return ret;
}

/* this function is no longer needed */
// py::object SumArrayT(np::ndarray arr1, np::ndarray arr2)
// {
//   np::arrayt<float> ret(np::empty(arr1.rank(), arr1.dims(), arr1.itemtype()));
//   
//   const np::ssize_t *dims = arr1.dims();
//   for (int i=0; i<dims[0]; ++i)
//   {
//     ret(i) = arr1(i) + arr2(i);
//   }
//   return ret.getObject();
// }



py::object SumNumericArray(np::ndarray arr1, np::ndarray arr2)
{
  //nm::array ret = py::extract<nm::array>(np::empty(1, &len, np::getItemtype(arr1)));
  //const Py_ssize_t len = py::extract<int>(py::getattr(arr1, "shape")[0]);
  //const Py_ssize_t len = arr1.get_nd();
  if( !(arr1.get_nd() == 1) )
  {
    throw std::invalid_argument("1d array expected");
  }
  py::tuple shape = py::make_tuple(arr1.shape(0));
  //np::dtype dtype = np::dtype::get_builtin<float>();
  np::ndarray ret = np::empty(shape,arr1.get_dtype());
  
  for (int i=0; i<arr1.shape(0); ++i)
  {
    float x1 = py::extract<float>(arr1[i]);
    float x2 = py::extract<float>(arr2[i]);
    ret[i] = py::object(x1 + x2);
  }
  return ret;
}



BOOST_PYTHON_MODULE(libdemo)
{
  Py_Initialize();
  cout<<"new boost numpy support detected"<<endl;
  np::initialize();
  py::def("printConvertedArray_int", printConvertedArray<int>);
//   py::def("SumArrayT", SumArrayT);
  py::def("printArray", printArray);
  py::def("int_to_str", scalar_to_str<int>);
  py::def("char_to_str", scalar_to_str<char>);
  py::def("float_to_str", scalar_to_str<float>);
  py::def("double_to_str", scalar_to_str<double>);
  py::def("SumNumericArray", SumNumericArray);
  py::def("ReturnedFromCPP", ReturnedFromCPP);
}


