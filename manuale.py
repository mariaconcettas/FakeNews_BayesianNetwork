import pandas as pd
import bnlearn as bn
import seaborn as sn
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

df = pd.read_csv('Datawithsentiment.csv',dtype={'label': object})

twenty_per = (len(df)*20)/100 #prendo il 20% del dataset
number=math.trunc(twenty_per) #prendo la parte intera
newsTest=df.iloc[0:number] #da 0 al 20% 
newsTrain=df.iloc[number:] #da 20% alla fine


dataTrain = pd.DataFrame(newsTrain, columns=["label","sentiment titolo","sentiment notizia","complessità"]) #dataset con le news da addestrare(80% del dataset)
dataTest = pd.DataFrame(newsTest, columns=["label","sentiment titolo","sentiment notizia","complessità"]) #dataset con le news da testare(20% del dataset)


#Definisco manualmente la struttura della rete
edges = ([
        ('sentiment titolo','label'),
        ('sentiment notizia','label'),
        ('complessità','label')
         ]) 


DAG = bn.make_DAG(edges)
bn.plot(DAG)

#per stimare i valori delle distribuzioni di probabilità condizionate(CPD).
model = bn.parameter_learning.fit(DAG, dataTrain)


def function_inference(data,model):            
    pred=[]
    for x in range(0,len(data)): 
        Evidence={}
        for y in range(1,len(data.columns)):
            Evidence.update({data.columns[y] : data[data.columns[y]][x]})
        print(Evidence)
        q = bn.inference.fit(model, variables=[data.columns[0]], evidence=Evidence)
        val=[q.state_names[data.columns[0]],q.values]         
        print(val)
        newdict = {}
        for i in range(len(val)):
            if i % 2 == 0:
                list1 = val[i] #chiave=label
                list2 = val[i+1] #valore=valore_label
                newdict = dict(zip(list1, list2))
                print(newdict)
                MaxKey = max(newdict, key=newdict.get)
                print(MaxKey)
        pred.append(MaxKey)
    print(pred)        
    print(len(pred)) 
    return pred           


pred_val=function_inference(dataTest,model)


y_true = dataTest[dataTest.columns[0]] #valori reali (colonna dell'etichette true/false del dataset)
y_pred = pred_val #valori previsti


#Costruisco la confusion matrix
ax= plt.subplot()
cf=confusion_matrix(y_true,y_pred)
print(cf)
group_counts = ["{0:0.0f}".format(value) for value in
                cf.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cf.flatten()/np.sum(cf)]
labels = [f"{v1}\n{v2}" for v1, v2 in
          zip(group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sn.heatmap(cf, annot=labels, fmt="", ax=ax)
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(["True","False"]); ax.yaxis.set_ticklabels(["True","False"])
plt.show()

#Accuracy and Classification Report
score=accuracy_score(y_true,y_pred)
print(f'Accuracy: {round(score*100,2)}%')
print(f"Classification Report : \n\n{classification_report(y_true, y_pred)}")
