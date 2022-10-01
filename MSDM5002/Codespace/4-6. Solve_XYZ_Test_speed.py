# -*- coding: utf-8 -*-
"""
@author: Junwei Liu
Test the speed of different functions by runing it repeatly
"""
import time
import numpy as np

def func_check(xx,yy,zz):
    get_result=0
    for x in xx:
        for y in yy:
            for z in zz:
                if x**3+y**3+z**3 == The_number:
                    get_result=1
                    return x,y,z,get_result    
    return x,y,z,get_result

#put the operations in the outer loop
def func_check1(xx,yy,zz):
    get_result=0
    for x in xx:
        x3=x**3
        for y in yy:
            y3=y**3
            for z in zz:
                if x3+y3+z**3 == The_number:
                    return x,y,z,get_result
    return x,y,z,get_result

#do the power calculations in advance since it will talke some time
def func_check2(xx,yy,zz):
    get_result=0
    xx3=[x**3 for x in xx]
    yy3=[y**3 for y in yy]
    zz3=[z**3 for z in zz]
    for x in xx3:
        for y in yy3:
            for z in zz3:
                if x+y+z == The_number:
                    get_result=1
                    return xx[xx3.index(x)],yy[yy3.index(y)],zz[zz3.index(z)],get_result
    return xx[xx3.index(x)],yy[yy3.index(y)],zz[zz3.index(z)],get_result

#replace the for loop by matrix operations
def func_check3(xx,yy,zz):

    xx3=[x**3 for x in xx]
    yy3=[y**3 for y in yy]
    zz3=[z**3 for z in zz]
    
    MX=np.array(xx3)
    MY=np.array(yy3)
    MZ=np.array(zz3)

    XY=np.zeros([len(xx),len(yy)])
    for n in range(len(yy)):
        XY[n,:]=MX+MY[n]
    
    XYZ=np.zeros([len(xx),len(yy),len(zz)])
    for n in range(len(zz)):  
        XYZ[n,:,:]=XY+MZ[n]
        
    T1,T2,T3=np.where(XYZ==The_number)
    
    if len(T1)>0:
        return xx[T1[0]],yy[T2[0]],zz[T3[0]],1
    else:
        return max(xx),max(yy),max(zz),0

#only keep the array which is necessary
def func_check4(xx,yy,zz):

    xx3=[x**3 for x in xx];     MX=np.array(xx3)

    XY=np.zeros([len(xx),len(yy)]) 
    for n in range(len(yy)):
        XY[n,:]=MX+yy[n]**3
        
    XYZ=np.zeros([len(xx),len(yy),len(zz)])
    for n in range(len(zz)):  
        XYZ[n,:,:]=XY+zz[n]**3
        
    T1,T2,T3=np.where(XYZ==The_number)
    
    if len(T1)>0:
        return xx[T1[0]],yy[T2[0]],zz[T3[0]],1
    else:
        return max(xx),max(yy),max(zz),0

#do the power calculations in advance
def func_check5(xx,yy,zz):

    xx3=[x**3 for x in xx];    MX=np.array(xx3)
    yy3=[y**3 for y in yy];    zz3=[z**3 for z in zz]
        
    XY=np.zeros([len(xx),len(yy)])
    for n in range(len(yy3)):
        XY[n,:]=MX+yy3[n]
        
    XYZ=np.zeros([len(xx),len(yy),len(zz)])
    for n in range(len(zz)):  
        XYZ[n,:,:]=XY+zz3[n]
        
    T1,T2,T3=np.where(XYZ==The_number)
    
    if len(T1)>0:
        return xx[T1[0]],yy[T2[0]],zz[T3[0]],1
    else:
        return max(xx),max(yy),max(zz),0

#use integer to replace float
def func_check6(xx,yy,zz):

    xx3=[x**3 for x in xx];    MX=np.array(xx3)
    yy3=[y**3 for y in yy];    zz3=[z**3 for z in zz]

    XY=np.zeros([len(xx3),len(yy3)],int)   
    for n in range(len(yy3)):
        XY[n,:]=MX+yy3[n]

    XYZ=np.zeros([len(xx3),len(yy3),len(zz3)],int)
    for n in range(len(zz3)):  
        XYZ[n,:,:]=XY+zz3[n]
        
    T1,T2,T3=np.where(XYZ==The_number)
    
    if len(T1)>0:
        return xx[T1[0]],yy[T2[0]],zz[T3[0]],1
    else:
        return max(xx),max(yy),max(zz),0

#change all the for loops to be array operations
def func_check7(xx,yy,zz):
    
    MX=np.array(xx)**3
    MY=np.array(yy)**3
    MZ=np.array(zz)**3
    
    XY=np.zeros([len(xx),len(yy)],int) 
    TT1=XY.copy()+MX
    TT2=XY.copy()+MY
    XY=TT1+TT2.transpose()

    XYZ=np.zeros([len(xx),len(yy),len(zz)],int)
    TT4=XYZ.copy()+XY
    TT5=XYZ.copy()+MZ
    XYZ=TT4+TT5.transpose()

    T1,T2,T3=np.where(XYZ==The_number)
    
    if len(T1)>0:
        return xx[T1[0]],yy[T2[0]],zz[T3[0]],1
    else:
        return max(xx),max(yy),max(zz),0

#remove some redundent statements
def func_check8(xx,yy,zz):
    
    MX=np.array(xx)**3
    MY=np.array(yy)**3
    MZ=np.array(zz)**3

    TT1=np.zeros([len(xx),len(yy)],int)+MX
    TT2=np.zeros([len(xx),len(yy)],int)+MY
    XY=TT1+TT2.transpose()

    TT4=np.zeros([len(xx),len(yy),len(zz)],int)+XY
    TT5=np.zeros([len(xx),len(yy),len(zz)],int)+MZ
    XYZ=TT4+TT5.transpose()

    T1,T2,T3=np.where(XYZ==The_number)
    
    if len(T1)>0:
        return xx[T1[0]],yy[T2[0]],zz[T3[0]],1
    else:
        return max(xx),max(yy),max(zz),0

#even fewer statements
def func_check9(xx,yy,zz):
    
    MX=np.array(xx)**3
    MY=np.array(yy)**3
    MZ=np.array(zz)**3
    
    XY=np.zeros([len(xx),len(yy)],int)+MX
    XY=XY.transpose()+MY
    
    XYZ=np.zeros([len(xx),len(yy),len(zz)],int)+XY
    XYZ=XYZ.transpose()+MZ

    T1,T2,T3=np.where(XYZ==The_number)
      
    if len(T1)>0:
        return xx[T1[0]],yy[T2[0]],zz[T3[0]],1
    else:
        return max(xx),max(yy),max(zz),0

def func_check10(xx,yy,zz):
    
    MX=np.array(xx)**3
    MY=np.array(yy)**3
    MZ=np.array(zz)**3
    
    # utilize broadcasting to implement outer addition
    XY=MX[...,np.newaxis]+MY
    # XY[:,np.newaxis] bug occur here, this expression add new axis in a wrong place.
    XYZ=XY[...,np.newaxis]+MZ

    T1,T2,T3=np.where(XYZ==The_number)
      
    if len(T1)>0:
        return xx[T1[0]],yy[T2[0]],zz[T3[0]],1
    else:
        return max(xx),max(yy),max(zz),0

def func_check11(xx,yy,zz):
    
    MX=np.array(xx)**3
    MY=np.array(yy)**3
    MZ=np.array(zz)**3
    
    XY=np.add.outer(MX,MY)
    XYZ=np.add.outer(XY,MZ)

    T1,T2,T3=np.where(XYZ==The_number)
      
    if len(T1)>0:
        return xx[T1[0]],yy[T2[0]],zz[T3[0]],1
    else:
        return max(xx),max(yy),max(zz),0

# The_number=13

# limit_test=100
# xx=[i for i in range(-limit_test,limit_test)];
# yy=[i for i in range(-limit_test,limit_test)];
# zz=[i for i in range(-limit_test,limit_test)];

# print("\nDo the test in range ["+str(-limit_test)+","+str(limit_test)+")") 
#start_T=time.time()
#print(func_check(xx,yy,zz))
#print("Time for func --- %s seconds ---"%(time.time()-start_T))
#start_T=time.time()
#print(func_check1(xx,yy,zz))
#print("Time for func1 --- %s seconds ---"%(time.time()-start_T)) 
#start_T=time.time()
#print(func_check2(xx,yy,zz))
#print("Time for func2 --- %s seconds ---"%(time.time()-start_T)) 
#start_T=time.time()
#print(func_check3(xx,yy,zz))
#print("Time for func3 --- %s seconds ---"%(time.time()-start_T)) 
#start_T=time.time()
#print(func_check4(xx,yy,zz))
#print("Time for func4 --- %s seconds ---"%(time.time()-start_T)) 
#start_T=time.time()
#print(func_check5(xx,yy,zz))
#print("Time for func5 --- %s seconds ---"%(time.time()-start_T)) 
#start_T=time.time()
#print(func_check6(xx,yy,zz))
#print("Time for func6 --- %s seconds ---"%(time.time()-start_T)) 
#start_T=time.time()
#print(func_check7(xx,yy,zz))
#print("Time for func7 --- %s seconds ---"%(time.time()-start_T)) 


The_number=12

limit_test=200
xx=[i for i in range(-limit_test,limit_test)];
yy=[i for i in range(-limit_test,limit_test)];
zz=[i for i in range(-limit_test,limit_test)];
print("\nDo the test in range ["+str(-limit_test)+","+str(limit_test)+")") 

test_num=10
T=np.zeros([12,test_num])
for n in range(test_num):
    # start_T=time.time(); func_check(xx,yy,zz);  T[0,n]=time.time()-start_T
    
    # start_T=time.time(); func_check1(xx,yy,zz); T[1,n]=time.time()-start_T
    
    # start_T=time.time(); func_check2(xx,yy,zz); T[2,n]=time.time()-start_T
    
    # start_T=time.time(); func_check3(xx,yy,zz); T[3,n]=time.time()-start_T

    # start_T=time.time(); func_check4(xx,yy,zz); T[4,n]=time.time()-start_T
    
    # start_T=time.time(); func_check5(xx,yy,zz); T[5,n]=time.time()-start_T
    
    start_T=time.time(); func_check6(xx,yy,zz); T[6,n]=time.time()-start_T
    
    # start_T=time.time(); func_check7(xx,yy,zz); T[7,n]=time.time()-start_T
    
    # start_T=time.time(); func_check8(xx,yy,zz); T[8,n]=time.time()-start_T
    
    start_T=time.time(); func_check9(xx,yy,zz); T[9,n]=time.time()-start_T

    start_T=time.time(); func_check10(xx,yy,zz); T[10,n]=time.time()-start_T

    start_T=time.time(); func_check11(xx,yy,zz); T[11,n]=time.time()-start_T

test_time=T.sum(axis=1)/test_num
print(test_time)

print('The number is',The_number)
print(6,func_check6(xx,yy,zz))
print(9,func_check9(xx,yy,zz))
print(10,func_check10(xx,yy,zz))
print(11,func_check11(xx,yy,zz))
