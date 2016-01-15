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