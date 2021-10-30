import pandas as pd
import bnlearn as bn
import numpy as np


df = pd.read_csv('newsDataset/True.csv', names=["titolo", "testo","soggetto", "data"],error_bad_lines=False)
df.insert(0, 'label', 'True', True)
print(df)
df1 = pd.read_csv('newsDataset/Fake.csv', names=["titolo", "testo","soggetto", "data"],error_bad_lines=False)
df1.insert(0, 'label', 'False', True)
df1 = df1.replace(['News'],'worldnews')
df1 = df1.replace(['politics'],'politicsNews')



frameTrue = pd.DataFrame(df,columns=["label","titolo", "testo","soggetto"])
frameFalse = pd.DataFrame(df1,columns=["label","titolo", "testo","soggetto"])

updateFalse=frameFalse[(frameFalse['soggetto'] != 'Government News') & (frameFalse['soggetto'] != 'Middle-east') & (frameFalse['soggetto'] != 'US_News') & (frameFalse['soggetto'] != 'left-news') ]

trueNews = frameTrue[frameTrue['label'] == 'True'].sample(n=5000) #prendo 5000 news vere a random dal dataset
falseNews = updateFalse[updateFalse['label'] == 'False'].sample(n=5000) #prendo 5000 news false a random dal dataset, dopo aver escluso alcune categorie (linea 19)

#inserisco tutte le news in un dataFrame
dataNewsTrue = pd.DataFrame(trueNews) 
dataNewsFalse = pd.DataFrame(falseNews)
resultData = dataNewsTrue.append(dataNewsFalse)

#mix di righe 
finalDataset = resultData.sample(frac=1, random_state=0)
print("lunghezza dataset:",len(finalDataset))

finalDataset.to_csv('newsDataset/dataNews.csv',index=False) #dataset con le 10.000 news = 5000 vere e 5000 false


