from TrackingDetection import TrackingDetection
import numpy as np

# Create a TrackingDetection object
data = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
mydata = TrackingDetection(data)

# Add some data 
mydata.update(1)
mydata.update(2)
mydata.update(3)
mydata.update(4)
print(mydata.get())
mydata.update(5)
mydata.update(6)
mydata.update(7)
mydata.update(8)
mydata.update(9)
mydata.update(10)
mydata.update(11)
mydata.update(12)
mydata.update(13)
mydata.update(14)
mydata.update(15)
mydata.print()
print(mydata.get())
mydata.update(16)
mydata.print()
print(mydata.get())
