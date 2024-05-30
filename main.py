import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

def mapeaza_varsta(ani):
    if pd.isna(ani):
        return None
    elif ani <= 20:
        return 1
    elif 20 < ani <= 40:
        return 2
    elif 40 < ani <= 60:
        return 3
    else:
        return 4

df = pd.read_csv(r'titanic\train.csv')

print("---------=Cerinta 1=---------")
 
print("Numarul de coloane este " + str(df.shape[1]))
for i in range(0,df.shape[1]):
    print(df.columns[i],end="  ")
print()
for i in range(0,df.shape[1]):
    print(df.dtypes.iloc[i],end="   ")

print("Coloanele cu valori lipsa")
print(df.isna().sum())

print("Numarul de linii este " + str(df.shape[0]))
print("Numarul de linii dublicate este " + str(len(df.to_string()) - len(df.drop_duplicates().to_string())))


print("---------=Cerinta 2=---------")

survivors = int(df["Survived"].mean()*100)
print("Procentul persoanelor care au supravietuit este " + str(survivors) + "%")

deads = 100 - survivors
print("Procentul persoanelor care nu au supravietuit este " +str(deads) + "%")

men = int(df["Sex"].eq("male").mean()*100)
print("Procentul barbatilor este " + str(men) + "%")

women = 100 - men
print("Procentul femeilor este " + str(women) + "%")
pclass = df.groupby('Pclass').size()/ df.groupby('Pclass').size().sum()*100
print("procentul pasagerilor pentru fiecare tip de clasă ")
print(pclass)

data = [survivors, deads,women,men,int(pclass[1]),int(pclass[2]),int(pclass[3])]
data_names = ["supravietuitori", "nesupravietuitori", "femei","barbati","clasa 1", "clasa 2", "clasa 3"]

# charturi pie
graph, pies = plt.subplots(1, 3)

pies[0].pie(data[0:2], labels=data_names[0:2], autopct='%1.1f%%')
pies[1].pie(data[2:4], labels=data_names[2:4], autopct='%1.1f%%')
pies[2].pie(data[4:], labels=data_names[4:], autopct='%1.1f%%')
plt.show()

print("---------=Cerinta 3=---------")

dir = 'titanic'
csvs = []

# big df cu toate fisierele din datasetu titanic concatenate
for file in os.listdir(dir):
        file_path = os.path.join(dir, file)
        csvs.append(file_path)

df_full = pd.DataFrame()

for file in csvs:
    cur_df = pd.read_csv(file)
    df_full = pd.concat([df_full, cur_df], ignore_index=True)
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]) and col != 'PassengerId':
        plt.figure()  
        df_full[col].dropna().hist(bins=20)  
        plt.title(f'Histograma {col}')  
        plt.ylabel('Frecvență')
        plt.xlabel(col)  
        plt.show()

print("---------=Cerinta 4=---------")

col_lipsa = df.isna().any()
df_col_lipsa = df.loc[:, col_lipsa]

print(df_col_lipsa)
surv = 0
dead = 0
for col in df_col_lipsa.columns:
    print(f"Pentru coloana {col} sunt {df_col_lipsa[col].isna().sum()} valori lipsa reprezentand {df_col_lipsa[col].isna().mean() * 100:.2f}%")
    surv = surv  + (df_col_lipsa[col].isna() & df['Survived']).mean() #facem & inte valorile nan care sunt true si coloana de survived 1 deci ne dau pers supravietuitoare cu incomplete
    dead = dead  + (df_col_lipsa[col].isna() & (df['Survived']==0)).mean() 
print()
print(f"Persoanele care au supravietuit dar aveau date incomplete reprezinta {surv*100:.2f}%")
print(f"Persoanele care nu au supravietuit dar aveau date incomplete reprezinta {dead*100:.2f}%")

print("---------=Cerinta 5=---------")
num_cat_varsta = [(df["Age"] <= 20).sum(),(((df["Age"] > 20)& (df["Age"]<=40)).sum()),((df["Age"] > 40)& (df["Age"]<=60)).sum(),(df["Age"] > 60).sum()] 
for i in range(0,4):
    print(f"Categoria {i+1} de varsta are {num_cat_varsta[i]} oameni")
df["Categorie_varsta"] = df['Age'].map(mapeaza_varsta) # a creat o coloana noua
df.to_csv('titanic/new_col.csv', index=False)
# Create a bar plot
cat_vs = np.array(["Categorie de varsta 1","Categorie de varsta 2","Categorie de varsta 3","Categorie de varsta 4"])
np_num_cat_varsta = np.array(num_cat_varsta)

# grafic bar
plt.figure(figsize=(10, 6))
plt.bar(cat_vs, np_num_cat_varsta)
plt.xlabel('Categorii de Vârstă')
plt.ylabel('Număr de Pasageri')
plt.title('Distribuția Pasagerilor pe Categorii de Vârstă')
plt.show()


print("---------=Cerinta 6=---------")

df_male = df[df['Sex'] == 'male']
df_male_survived = df_male[df_male["Survived"] == 1]
pclass = df_male_survived.groupby('Categorie_varsta').size()
print("Numarul barbatilor care au supravietuit in functie de varsta lor")
print(pclass)

plt.scatter(df_male['Age'], df_male['Survived'], alpha=0.5)
plt.title('Relatie varsta barbatie supravietuire')
plt.show()

print("---------=Cerinta 7=---------")
children = df[df["Age"]<18]
children_survived = children[children["Survived"]==1].shape[0]/children.shape[0]*100
print(f"Procentul copiilor aflati la bord este {children.shape[0]/df.shape[0]*100:.2f}%")
alive_adults = (df["Age"] >= 18 & df["Survived"]).mean()
vect = np.array(["rata de supravieturie copii","rata de supravieturie adulti"])
np_data_adults_vs_children = np.array([children_survived,alive_adults])

plt.bar(vect, np_data_adults_vs_children)
plt.ylabel('Procentaj')
plt.title('Distribuția Pasagerilor pe Categorii de Vârstă')
plt.show()


print("---------=Cerinta 8=---------")

# functie de medie 

def return_mean_survived(df, col):
    if 'Survived' in df.columns:
        mean_survived = df[df['Survived'] == 1][col].mean()
        mean_not_survived = df[df['Survived'] == 0][col].mean()
        df.loc[(df[col].isna()) & (df['Survived'] == 0), col] = mean_not_survived
        df.loc[(df[col].isna()) & (df['Survived'] == 1), col] = mean_survived
    else:  # dam fill pe baza 'Sex' daca 'Survived' nu exista in caz ca testam cu test.csv unde nue xista survived
        if 'Sex' in df.columns:
            mean_male = df[df['Sex'] == 'male'][col].mean()
            mean_female = df[df['Sex'] == 'female'][col].mean()
            df.loc[(df[col].isna()) & (df['Sex'] == 'female'), col] = mean_female
            df.loc[(df[col].isna()) & (df['Sex'] == 'male'), col] = mean_male
    return df

for col in df_col_lipsa.columns:
    if df_col_lipsa[col].isna().any():
        if pd.api.types.is_numeric_dtype(df_col_lipsa[col]):
            return_mean_survived(df, col)
        else:
            most_frequent_value = df[col].mode()[0]
            df[col] =df[col].fillna( most_frequent_value)

df.to_csv('Date/task8.csv', index=False)


print("---------=Cerinta 9=---------")


titluri = {
    'Mrs.': 'female','Sir': 'male',
    'Gen.': 'male','Ms.': 'female',
    'Dona': 'female','Capt.': 'male',
    'Judge': 'any','Rev.': 'any',
    'Master': 'male','Dr.': 'any',
    'Countess': 'female','Mr.': 'male',
    'Lady': 'female','Major': 'male',
    'Col.': 'male','Mx.': 'any',
    'Miss': 'female','Don': 'male'
}
def verif_titluri(row):
    titlu = row['Titluri']
    sex = row['Sex']
    if titlu in titluri:
        sexx = titluri[titlu]
        return sexx == 'any' or sexx == sex
    return False  

dff = df #copie dataframe cu col de titluri adaugata
dff["Titluri"] = df['Name'].apply(lambda x: x.split(',')[1].split()[0])
dff['Sex_Match'] = dff.apply(verif_titluri, axis=1)

print("Verificatre dacă titlurile de noblete regăsite în coloana Name (Mr., Mrs., Don, etc.) corespund cu sexul persoanei respective")
print(dff['Sex_Match'])

title_counts = df['Titluri'].value_counts()

plt.figure(figsize=(10, 6))
title_counts.plot(kind='bar')
plt.title('Câte persoane corespund fiecărui titlu')
plt.ylabel('Nr persoane')
plt.xlabel('Titlu')
plt.show()

print("---------=Cerinta 10=---------")
# Utilizăm Seaborn pentru a crea un catplot
data_h =df.head(100)
sns.catplot(x="Pclass", y="Fare", hue="Survived", data=data_h, kind="swarm", height=6, aspect=3)
plt.title('Relația dintre Tarif, Clasă și Supraviețuire(100 inregistrari)')
plt.xlabel('PClass')
plt.ylabel('Tarif')
plt.show()