# Copyright (c) 2016 Michael Welter
#
# This file is part of numpycpp.
#
# numpycpp is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# numpycpp is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with numpycpp.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import sys
import os

sys.path.append(os.getcwd())
import libdemo

print ('VARIOUS TYPES MAPPING:')
datatypes = [np.object, np.int8, np.bool, np.byte, np.short, np.int16, np.int, np.long, np.int32, np.int64, np.longlong, np.float, np.float32, np.double, np.float64]
for dt in datatypes:
  a = np.zeros(5, dtype = dt)
  print ('-----' + str(dt) + '-------')
  libdemo.printArray(a, False)
  
print ('')
print ('UNSIGNED TYPES:')
unsignedtypes = [np.uint8, np.ubyte, np.ushort, np.uint16, np.uint, np.uint32, np.uint64, np.ulonglong]
for dt in unsignedtypes:
  a = np.zeros(5, dtype = dt)
  print ('-----' + str(dt) + '-------')
  libdemo.printArray(a, False)

print ('PRINTING CONTENTS:')
a = np.arange(2*3*4, dtype = np.int32).reshape(2,3,4)
print ('PRINT IN PYTHON:')
print (a)
print ('PRINT c++ DEMO:')
libdemo.printArray(a, True)

a = np.asarray(['hello', 'python', 'wrapper!'], dtype = np.object)
print ('PRINT IN PYTHON:')
print (a)
print ('PRINT c++ DEMO:')
libdemo.printArray(a, True)


print ('PRINTING CONVERTED ARRAY:')
a = np.arange(2*3*4, dtype = np.int32).reshape(2,3,4)
libdemo.printConvertedArray_int(a)


def wrap_to_str(f):
  def wrapper(x):
    try:
      return f(x)
    except:
      return 'ERR'
  return wrapper

for fun in [libdemo.int_to_str, libdemo.char_to_str, libdemo.float_to_str, libdemo.double_to_str]:
  locals()[fun.__name__] = wrap_to_str(fun)

print ('TESTING SCALAR CONVERSION:')
types = [np.int8, np.byte, np.short, np.int16, np.int, np.long, np.int32, np.int64, np.longlong, np.uint8, np.ubyte, np.ushort, np.uint16, np.uint, np.uint32, np.uint64, np.ulonglong, np.float, np.float32, np.double, np.float64]
for dt in types:
  a = np.asarray((42,), dtype = dt)[0]
  print ('%s: as int: %s, as char: %s, as float: %s, as double: %s' % (str(dt), int_to_str(a), char_to_str(a), float_to_str(a), double_to_str(a)))

print ('RETURNING ARRAY FROM CPP:')
a = libdemo.ReturnedFromCPP()
print a


if 0:
  # direct memory access with arrayt<float> is approximately 38x faster than using numeric::array
  print ('PERFORMANCE TEST:')
  import time
  import matplotlib.pyplot as pyplot
  
  def PerfTest(fun):
    print ('TEST: '+str(fun.__name__))
    sizes = [1000, 10000, 100000, 1000000]
    times = []
    for s in sizes:
      a1 = np.arange(s, dtype = np.float32)
      a2 = np.ones(s, dtype = np.float32)
      tavg = []
      for n in xrange(10):
        t0 = time.clock()
        result = fun(a1, a2)
        t1 = time.clock()
        tavg.append(t1 - t0)
      times.append(np.average(tavg))
    return sizes, np.asarray(times)
  
  sz, perfArrayT     = PerfTest(libdemo.SumArrayT)
  print perfArrayT
  sz, perfBoostArray = PerfTest(libdemo.SumNumericArray)
  print perfBoostArray
  
  pyplot.plot(sz, perfBoostArray / perfArrayT)
  pyplot.gca().set(xlabel = 'Array Size', ylabel = 'Relative Compute Time')
  pyplot.show()
