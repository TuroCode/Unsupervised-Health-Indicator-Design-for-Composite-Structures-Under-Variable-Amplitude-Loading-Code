import numpy as np
import matplotlib.pyplot as plt
from PMT_Evaluation import PMT_report

#Import the numpy array of features (from test set)

#matrixes_columns=np.load("Feature_matrix_test_2_45.npy") #Columns of the raw data of CSS,CCNT,R,D,RMS
matrixes=np.load("Feature_matrix.npy")
matrixes=matrixes[[5,16,7,18,14]]
coeffs=np.array([0.09632068,0.83594863,0.79086729,-0.28853425,-0.09189289])
fit_coeff=np.array([0.599999,0.4,0.00000001])
row_names=["45% no1","60% no2","60% no3","45% no4","60% no4","45% no5","var loading 1","var loading 3","var loading 4","var loading 5","var loading 6","var loading 8"]
color_arr=["lightgreen","cyan","cadetblue","lawngreen","tab:blue","tab:green","slateblue","darkslateblue","mediumslateblue","mediumpurple","rebeccapurple","blueviolet"]
color_arr_comp=["tab:green","tab:red","tab:blue","tab:orange","tab:purple","tab:brown"]


def HI_fun(matrixes,coeff):
    val=np.zeros((matrixes[0].shape[0],matrixes[0].shape[1]))

    "Definiton of the HI, number of features and defintion are free to be changed"
    for i in range(coeff.shape[0]):
        val=val+matrixes[i]*coeff[i]
    
    return val

HI=HI_fun(matrixes,coeffs) #Attains HI from the coefficients and data given
PMT_array=[0,0,0]
#Evaluation of PMT of the reaulting HI
Fitness=PMT_report(HI,125,fit_coeff,Fit=True)
PMT_array=PMT_report(HI,125,fit_coeff,Fit=False)



#Plotting of HI on all test files, and plot them then per case scenario on subfigures

feature=[0,1,2,3,4,5,6,7,8,9,10] #Actually files...

plt.subplot(221)
for i,row in enumerate(HI[feature]):
    row_drop=row[row!=0]
    plt.scatter(len(row_drop),row_drop[-1],marker="x",color=color_arr[i])
    plt.plot(np.arange(1,len(row_drop)+1),row_drop,color=color_arr[i],label=row_names[feature[i]])
    plt.xlabel("Window")
    plt.ylabel("HI Value")

#plt.plot(np.arange(850),np.arange(850)*0+1.5,label="Proposed Thershold",linestyle="dashed",alpha=0.5,color="r")
plt.legend()


plt.subplot(222)
feature=[0,3,5]

for i,row in enumerate(HI[feature]):
    row_drop=row[row!=0]
    plt.scatter(len(row_drop),row_drop[-1],marker="x",color=color_arr[feature[i]])
    plt.plot(np.arange(1,len(row_drop)+1),row_drop,color=color_arr[feature[i]],label=row_names[feature[i]])
    plt.xlabel("Window")
    plt.ylabel("HI Value")
plt.legend()


plt.subplot(223)
feature=[1,2,4]

for i,row in enumerate(HI[feature]):
    row_drop=row[row!=0]
    plt.scatter(len(row_drop),row_drop[-1],marker="x",color=color_arr[feature[i]])
    plt.plot(np.arange(1,len(row_drop)+1),row_drop,color=color_arr[feature[i]],label=row_names[feature[i]])
    plt.xlabel("Window")
    plt.ylabel("HI Value")
plt.legend()


plt.subplot(224)
feature=[6,7,8,9,10,11]

for i,row in enumerate(HI[feature]):
    row_drop=row[row!=0]
    plt.scatter(len(row_drop),row_drop[-1],marker="x",color=color_arr[feature[i]])
    plt.plot(np.arange(1,len(row_drop)+1),row_drop,color=color_arr[feature[i]],label=row_names[feature[i]])
    plt.xlabel("Window")
    plt.ylabel("HI Value")
plt.legend()

plt.show()



##Compare 45% no4 with components that make it up


row=HI[3]
row=row[row!=0] #Clear zeroes



legend_arr=['std(CSS)','entropy(CCNT)','std(R)','entropy(D)','maxima(RMS)']

for i,feature in enumerate(matrixes):
    feature=feature[3,:]
    feature=feature[feature!=0]
    print(color_arr[i])
    
    plt.plot(range(len(feature)),feature,linewidth="1",color=color_arr_comp[i],alpha=0.4,label=legend_arr[i])
    plt.scatter(len(feature),feature[-1],color=color_arr[i],marker="x",alpha=0.4)

plt.plot(range(len(row)),row,linewidth="1.2",color='dimgrey',label="HI")
plt.scatter(len(row),row[-1],marker="x",color="dimgrey")
plt.xlabel("Window")
plt.legend()
plt.show()






#Plotting of PMT of raw features, PMT of extracted features and PMT of HI

x_type=np.array(["Raw features","Extracted features","Optimised HI"])
x=np.arange(len(x_type))
features_fitness_used=[]
features_monotonicity=[]
features_prognosability=[]

for feature in matrixes:
    features_fitness_used.append(abs(PMT_report(feature,125,fit_coeff,Fit=True)))
    P,M,T=PMT_report(feature,125,fit_coeff)
    
    features_prognosability.append(P)
    features_monotonicity.append(abs(M))
mean=np.mean(features_fitness_used)

#Monotonicity related
mean_monot=np.mean(features_monotonicity)
upper_err_monot=np.max(features_monotonicity)-mean_monot
lower_err_monot=mean_monot-np.min(features_monotonicity)

mean_prognos=np.mean(features_prognosability)
upper_err_prognos=np.max(features_prognosability)-mean_prognos
lower_err_prognos=mean_prognos-np.min(features_prognosability)

#Took data from Katerina's baseline analysis
#['std(CSS)','entropy(CCNT)','std(R)','entropy(D)','maxima( RMS)']
raw_fitness=np.array([0.3215,0.2978,0.2318,0.3643,0.2501])
raw_monoton=np.array([0.2112,0.2141,0.1619,0.1820,0.1892])
raw_prognos=np.array([0.4869,0.4232,0.3366,0.6378,0.9089])
mean_raw=np.mean(raw_fitness) #Mean fitness from the raw columns


bwidth=0.25

y_prognos=[np.mean(raw_prognos),mean_prognos,PMT_array[0]]
y_monton=[np.mean(raw_monoton),mean_monot,PMT_array[1]]
y_fit=[mean_raw,mean,Fitness]
x_prognos=x-bwidth
x_fit=x+bwidth



#error compuation st the maximumand minimum values are shown in the bar chart
c_fit=[[mean_raw-min(raw_fitness),mean-min(features_fitness_used),0],[max(raw_fitness)-mean_raw,max(features_fitness_used)-mean,0]]
c_monot=[[np.mean(raw_monoton)-np.min(raw_monoton),lower_err_monot,0],[np.max(raw_monoton)-np.mean(raw_monoton),upper_err_monot,0]]
c_prognos=[[np.mean(raw_prognos)-np.min(raw_prognos),lower_err_prognos,0],[np.max(raw_prognos)-np.mean(raw_prognos),upper_err_prognos,0]]


#Plotting of bars, want to plt next ot one another...

plt.bar(x,y_monton,edgecolor="black",color="tab:blue",width=bwidth,label="Absolute Monotonicity")
plt.bar(x_prognos,y_prognos,edgecolor="black",color="tab:orange",width=bwidth,label="Prognosability")
plt.bar(x_fit,y_fit,edgecolor="black",color="tab:green",width=bwidth,label="Fitness")

plt.xticks(x,x_type)#Basically re_names the x axis
alpha_gen=0.6
cap=10
plt.errorbar(x,y_monton,c_monot,fmt='o',capsize=cap,color="tab:red",alpha=alpha_gen)
plt.errorbar(x_prognos,y_prognos,c_prognos,fmt='o',capsize=cap,color="tab:red",alpha=alpha_gen)
plt.errorbar(x_fit,y_fit,c_fit,fmt='o',capsize=cap,color="tab:red",alpha=alpha_gen)

plt.legend()

plt.show()
