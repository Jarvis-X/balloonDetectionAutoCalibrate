from TrackingDetection import TrackingDetection
import numpy as np

# Create a TrackingDetection object
data = [[0] * 10 for _ in range(10)]
print(len(data))
mydata = [None] * len(data)
print(len(mydata))
for i in range(len(data)):
    mydata[i] = TrackingDetection(data[i])

# Add some data 
# mydata[1].update(1)
# mydata[1].update(2)
# mydata[1].update(3)
# mydata[1].update(4)
# print("data 1 average")
# print(mydata[1].get())
# print("data 1")
# mydata[1].print()
# mydata[1].update(5)
# mydata[1].update(6)
# mydata[1].update(7)
# mydata[2].update(8)
# mydata[2].update(9)
# mydata[2].update(10)
# mydata[2].update(11)
# mydata[2].update(12)
# mydata[2].update(13)
# mydata[2].update(14)
# mydata[2].update(15)
# print("data 1")
# mydata[1].print()
# print("data 2")
# mydata[2].print()
# print("data 2 average")
# print(mydata[2].get())
# mydata[1].update(16)
# print("data 1")
# mydata[1].print()
# print("data 1 average")
# print(mydata[1].get())

x = [0] * len(data)
print(x)
x[0] = 1
print(x)
