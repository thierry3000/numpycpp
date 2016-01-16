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

print ('TESTING SCALAR CONVERSION:')
inttypes = [np.int8, np.byte, np.short, np.int16, np.int, np.long, np.int32, np.int64, np.longlong, np.uint8, np.ubyte, np.ushort, np.uint16, np.uint, np.uint32, np.uint64, np.ulonglong]
for dt in inttypes:
  a = np.asarray((42,), dtype = dt)[0]
  print dt,':',
  libdemo.print_int(a)
  print ''

floattypes = [np.float, np.float32, np.double, np.float64]
for dt in floattypes:
  a = np.asarray((42,), dtype = dt)[0]
  print dt,':',
  libdemo.print_float(a)
  print ',',
  libdemo.print_double(a)
  print ''