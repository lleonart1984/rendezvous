import PIL
import numpy as np
import matplotlib.pyplot as plt

image = PIL.Image.open("./tests/models/wood.jpg")
width, height = image.size
image = image.convert("RGBA")
data = image.getdata()
bytes = bytearray()
for r,g,b,a in data:
    bytes.append(r)
    bytes.append(g)
    bytes.append(b)
    bytes.append(a)
im_np = np.array(bytes, dtype=np.uint8)

im_np = im_np.reshape((height, width, 4))
plt.imshow(im_np)
plt.show()