import numpy as np

### Array generation with arange and linspace
# start = 0
# stop = 10
# steparray = np.arange(start, stop, step = 1)
# intervalarray = np.linspace(start, stop, num=10, endpoint = False, dtype = int)
# print(steparray)
# print(intervalarray)

# decreasing = np.arange(stop, start, step = -1)
# print(decreasing)


### Basic operations
# x = np.array([13,24,21.2,17.6,21.7],'float')
# print(x.sum(),x.mean(),x.std(), sep='\n')


### Array slicing
# ax = np.array([1,3,5,7,9])
# print(ax[2:4])
# print(ax[2:])
# print(ax[:4])
# print(ax[:])
# print(ax[:6]) # no errors!
# print(ax[4:2]) # empty list


### Negative indexing
# foo = np.arange(start = 0, stop = 12, step = 2)
# print(foo)
# print(foo[-1])
# print(foo[-4:-2])
# print(foo[-2:-4]) # empty list


### NumPy array copying
# # this points to the same memory address as ax[].
# # sax = ax[:]
# # this creates a true copy of ax[].
# sax = np.copy(ax)
# ax[0] = 2
# print(f"Ax: {ax}")
# print(f"Sax: {sax}")

### Boolean masks
# names = np.array(['Bill','Mike','Tom','Kathy','Giovanni','Catherine'])
# bonus = np.array([232300.56,478123.45,3891.24,98012.36,52123.50,0])
# print(names[bonus > 130000])


### Multi-dimensional arrays 
nonsingular = np.array([
                        (1, 0), 
                        (1, 1)
                        ])

# parentheses types are interchangable for multi-dimensional arrays.
nonsingular2 = np.array([
                        [1, 0], 
                        (1, 1)
                        ], dtype = int)    
# array dimensions must be consistent. This will raise a ValueError:
# invalidmatrix = np.array([ (1, 0), (0) ])

# automatic type casting occurs for mixed types.
validmatrix = np.array([
                        (1, 0),
                        ('we are', "strings!")
                        ])

print(f"{nonsingular}\n{nonsingular2}\n{validmatrix}")

c = np.array([
        [(1.5,2,3), (4,5,6)],
        [(3,2,1), (4,5,6)]
            ], dtype = float)

