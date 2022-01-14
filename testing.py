

class CounterInstance(type):
    def __new__(cls, name, bases, dict):
        cls.counter = 0
        return super().__new__(cls, name, bases, dict)

    def __call__(cls, *args, **kwarg):
        cls.counter += 1
        return super().__call__(*args, **kwarg)

class A(metaclass=CounterInstance):
    pass

class C(A):
    pass


a1 = A()
a2 = A()
c1 = C()

print(A.counter)
print(C.counter)


# import PIL
# import numpy as np
# import matplotlib.pyplot as plt
#
# image = PIL.Image.open("./tests/models/wood.jpg")
# width, height = image.size
# image = image.convert("RGBA")
# data = image.getdata()
# bytes = bytearray()
# for r,g,b,a in data:
#     bytes.append(r)
#     bytes.append(g)
#     bytes.append(b)
#     bytes.append(a)
# im_np = np.array(bytes, dtype=np.uint8)
#
# im_np = im_np.reshape((height, width, 4))
# plt.imshow(im_np)
# plt.show()