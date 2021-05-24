

import matplotlib.pyplot as plt 
  
# line 1 points 
x1 = [0,1] 
y1 = [0,872.30] 
# plotting the line 1 points  
plt.plot(x1, y1, label = "AXGB ensemble") 
  
# line 2 points 
x2 = [0,1] 
y2 = [0,8333] 
# plotting the line 2 points  
plt.plot(x2, y2, label = "AXGB incremental") 

# line 3 points 
x3 = [0,1] 
y3 = [0,359.35] 
# plotting the line 2 points  
plt.plot(x3, y3, label = "AXGB thread") 
  
# naming the x axis 
plt.xlabel('Tempo') 
# naming the y axis 
plt.ylabel('Instâncias') 
# giving a title to my graph 
plt.title('Two lines on same graph!') 
  
# show a legend on the plot 
plt.legend() 
  
# function to show the plot 
plt.show() 