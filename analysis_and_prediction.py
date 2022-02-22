import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.svm import SVC, LinearSVC
#%% ucitavanje baze u DataFrame
df = pd.read_csv("recipes.csv")
#%% koliko ima obelezja (kolona) i uzoraka (vrsta)
print("shape: \n", df.shape)
# imamo 10566 uzoraka i 152 obelezja
#%% nedostajuci podaci
NANs = df.isnull().sum()
print("\n Nedostajuće vrednosti obeležja: \n", NANs)
udeo = df.isnull().sum()/len(df)*100
print("\n Procenat nedostajućih vrednosti obeležja: \n", udeo)
# nema nedostajucih vrednosti
#%% da li postoje nelogicne i nevalidne vrednosti
describe_tabela=df.describe()
# nema nevalidnih i nelogicnih vrednosti
#%% izbacivanje obelezja
df.drop(['Unnamed: 0'], inplace=True, axis=1)
# izbaceno je obelezje unnamed zato sto nije potrebno za analizu
#%% ispis svih drzava za koje imamo recepte
print("Drzave cije recepte analiziramo su: \n", df['country'].unique())
#%% ipis koliko koja drzava ima recepata
df_s=(df[df['country']=='southern_us']).count().unique()
df_f=(df[df['country']=='french']).count().unique()
df_g=(df[df['country']=='greek']).count().unique()
df_m=(df[df['country']=='mexican']).count().unique()
df_i=(df[df['country']=='italian']).count().unique()
df_j=(df[df['country']=='japanese']).count().unique()
df_c=(df[df['country']=='chinese']).count().unique()
df_t=(df[df['country']=='thai']).count().unique()
df_b=(df[df['country']=='british']).count().unique()
print("Southern US ima" , df_s, "recepata, Francuska ima", df_f, "recepata, Grcka ima",
      df_g, "recepata, Meksiko ima", df_m, "recepata, Italija ima", df_i, "recepata, Japan ima",
      df_j, "recepata, Kina ima", df_c, "recepata, Thai ima", df_t, "recepata, Engleska ima", df_b, "recepata.")
#%% prikaz koliko koja drzava ima recepata
country_column=df.iloc[:,-1]
df.rename(columns={"country":"recipes_no"},inplace=True)
df_new=df.groupby(by=country_column).count().reset_index()

plt.barh(df_new["country"],df_new["recipes_no"],color="green")
#plt.tick_params(axis='x', rotation=20)
plt.xlabel('Number of recipes')
plt.ylabel('Country')
plt.title('Number of recipes for each country')

#%% tabela sa svim drzavama
df_country = pd.DataFrame(df['recipes_no'].unique())
#%% prikaz koliko sastojaka koja drzava ukupno koristi, za svaki recept
ingredient_sum = df.groupby(by = country_column).sum().reset_index()
#%% sastojci korisceni po regionima
ingredient_max = ingredient_sum.select_dtypes(np.float64).idxmax(axis = 1)
ingredient_min = ingredient_sum.select_dtypes(np.float64).idxmin(axis = 1)

print("Najvise se koriste sastojci u sledecem redosledu:\n", ingredient_max)
print("\nNajmanje se koriste sastojci u sledecem redosledu:\n", ingredient_min)
ingridient_max_min = pd.concat([df_country,ingredient_max,ingredient_min], axis = 1)
#%% ispis 10 najcesce koriscenih sastojaka
df_ingredient = df.sum(axis = 0).reset_index()
df_ingredient_10 = df_ingredient[0:10]
print("\n10 najcesce koriscenih sastojaka", df_ingredient_10)
#%% prikaz 10 najcesce koriscenih sastojaka 
plt.barh(df_ingredient_10['index'],df_ingredient_10[0],color="orange")
plt.tick_params(axis='x', rotation=20)
plt.ylabel('Ingredients')
plt.xlabel('Number')
plt.title('Number of 10 most used ingredients')
#%% izdvanjanje vina kao sastojak
wine = df[['recipes_no','white wine','dry white wine','red wine','wine','wine vinegar']]
#%% recepti u kojima se dodaje vino
wine_columns = wine[(wine['white wine']!=0)|(wine['dry white wine']!=0)|(wine['red wine']!=0)|(wine['wine']!=0)|(wine['wine vinegar']!=0)]
#1434 recepata dodaje vino u jelo
#%% ispis procenta recepata koji sadrze vino
print("Procenat recepata koji koriste vino kao sastojak", 1434/10566*100)
#%% grupisanje recepata po drzavama 
df_wine = wine_columns.groupby(by = country_column).count().reset_index()
#%% prikaz broja recepata sa vinom
plt.barh(df_wine["country"],df_wine["recipes_no"], color="pink")
plt.xlabel('Number of recipes')
plt.ylabel('Country')
plt.title('Number of recipes using wine, for each country')
plt.tick_params(axis='x', rotation=20)
#%% izdvanjanje ljutih zacina kao sastojak
hot_spices = df[['recipes_no','pepper','black pepper','ground black pepper','red pepper','chili','ground pepper','chili powder','cayenne pepper','jalapeno chilies','pepper flakes','red pepper flakes','white pepper','freshly ground pepper']]
#%% recepti u kojima se dodaje ljute zacine
hot_spices_columns = hot_spices[(hot_spices['pepper']!=0)|(hot_spices['black pepper']!=0)|(hot_spices['ground black pepper']!=0)|(hot_spices['red pepper']!=0)|(hot_spices['chili']!=0)|
                                (hot_spices['ground pepper']!=0)|(hot_spices['chili powder']!=0)|(hot_spices['cayenne pepper']!=0)|(hot_spices['jalapeno chilies']!=0)|(hot_spices['pepper flakes']!=0)
                                |(hot_spices['red pepper flakes']!=0)|(hot_spices['white pepper']!=0)|(hot_spices['freshly ground pepper']!=0)]
#10566 recepata dodaje ljute zacine u jelo
#%% ispis 3 najcesce koriscena ljutih zacina
df_hot_spices = hot_spices_columns.sum(axis = 0).reset_index()
df_hot_spices_iloc = df_hot_spices.iloc[1: , :]
df_hot_spices_3 = df_hot_spices_iloc[0:3]
print("3 najcesce koriscenih ljutih zacina", df_hot_spices_3)
#%% prikaz svih recepata koji koriste ljute zacine
hot_spices_columns_sum = hot_spices_columns.groupby(['recipes_no']).sum().reset_index()
hot_spices_columns_sum.set_index('recipes_no').plot.bar()
plt.ylabel('Number of recipes')
plt.xlabel('Country')
plt.title('Number of recipes using hot spices, for each country')
plt.tick_params(axis='x', rotation=20)
plt.rcParams["figure.figsize"] = (10,5)
#%%                         KNN klasifikator

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print('nedostajućih vrednosti ima: ', X.isnull().sum().sum())
print('oznake klasa su: ', y.unique())
print('uzoraka u prvoj klasi ima: ', sum(y=='southern_us'))
print('uzoraka u drugoj klasi ima: ', sum(y=='french'))
print('uzoraka u trecoj klasi ima: ', sum(y=='greek'))
print('uzoraka u cetvrtoj klasi ima: ', sum(y=='mexican'))
print('uzoraka u petoj klasi ima: ', sum(y=='italian'))
print('uzoraka u sestoj klasi ima: ', sum(y=='japanese'))
print('uzoraka u sedmoj klasi ima: ', sum(y=='chinese'))
print('uzoraka u osmoj klasi ima: ', sum(y=='thai'))
print('uzoraka u devetoj klasi ima: ', sum(y=='british'))
#%%
y_class=y.groupby(by=y).count()
# primetimo da klase nisu jednako zastupljene, uzorci po klasama variraju
x_class=X.groupby(by=y).describe()
# podela podataka na trening i test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=10, stratify=y)
#%% definisati funkciju koja na osnovu matrice konfuzije računa tačnost klasifikatora po klasi i prosečnu tačnost
def tacnost_po_klasi(mat_konf, klase):
    tacnost_i = []
    N = mat_konf.shape[0]
    for i in range(N):
        j = np.delete(np.array(range(N)),i) 
        TP = mat_konf[i,i]
        F = 0
        F = (sum(mat_konf[i,j]) + sum(mat_konf[j,i]))
        TN = sum(sum(mat_konf)) - F - TP
        tacnost_i.append((TP+TN)/sum(sum(mat_konf)))
        print('Za klasu ', klase[i], ' tacnost je: ', tacnost_i[i])
    tacnost_avg = np.mean(tacnost_i)
    return tacnost_avg
#%% definisati funkciju koja na osnovu matrice konfuzije računa osetljivost klasifikatora po klasi, kao i prosečnu, odnosno makro osetljivost
def osetljivost_po_klasi(mat_konf, klase):
    osetljivost_i = []
    N = mat_konf.shape[0]
    for i in range(N):
        j = np.delete(np.array(range(N)),i) 
        TP = mat_konf[i,i]
        FN = sum(mat_konf[i,j])
        osetljivost_i.append(TP/(TP+FN))
        print('Za klasu ', klase[i], ' osetljivost je: ', osetljivost_i[i])
    osetljivost_avg = np.mean(osetljivost_i)
    return osetljivost_avg
#%% definisati funkciju koja na osnovu matrice konfuzije računa preciznost klasifikatora po klasi, kao i prosečnu, odnosno makro preciznost
def preciznost_po_klasi(mat_konf, klase):
    osetljivost_i = []
    N = mat_konf.shape[0]
    for i in range(N):
        j = np.delete(np.array(range(N)),i)
        TP = mat_konf[i,i]
        FP = sum(mat_konf[j,i])
        osetljivost_i.append(TP/(TP+FP))
        print('Za klasu ', klase[i], ' preciznost je: ', osetljivost_i[i])
    osetljivost_avg = np.mean(osetljivost_i)
    return osetljivost_avg
#%%
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
acc = []
for k in [1, 5, 10]:
    for m in ['jaccard', 'dice']:
        indexes = kf.split(X_train, y_train)
        acc_tmp = []
        fin_conf_mat = np.zeros((len(np.unique(y)),len(np.unique(y_train))))
        for train_index, test_index in indexes:
            classifier = KNeighborsClassifier(n_neighbors=k, metric=m)
            classifier.fit(X_train.iloc[train_index,:], y_train.iloc[train_index])
            y_pred = classifier.predict(X_train.iloc[test_index,:])
            acc_tmp.append(accuracy_score(y_train.iloc[test_index], y_pred))
            fin_conf_mat += confusion_matrix(y_train.iloc[test_index], y_pred, labels=y_train.unique())
        print('za parametre k=', k, ' i m=', m, ' tacnost je: ', np.mean(acc_tmp), ' a mat. konf. je:')
        #print(fin_conf_mat)

        disp = ConfusionMatrixDisplay(confusion_matrix =fin_conf_mat,  display_labels=classifier.classes_)
        disp.plot(cmap="Blues", values_format='', xticks_rotation=90)  
        plt.show()
        
        acc.append(np.mean(acc_tmp))
print('najbolja tacnost je u iteraciji broj: ', np.argmax(acc))
#%% 
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
acc = []
indexes = kf.split(X_train, y_train)
acc_tmp = []
fin_conf_mat = np.zeros((len(np.unique(y)),len(np.unique(y_train))))
for train_index, test_index in indexes:
    classifier = KNeighborsClassifier(n_neighbors=k, metric=m)
    classifier.fit(X_train.iloc[train_index,:], y_train.iloc[train_index])
    y_pred = classifier.predict(X_train.iloc[test_index,:])
    acc_tmp.append(accuracy_score(y_train.iloc[test_index], y_pred))
    fin_conf_mat += confusion_matrix(y_train.iloc[test_index], y_pred, labels=y_train.unique())
        #print(fin_conf_mat)

    disp = ConfusionMatrixDisplay(confusion_matrix =fin_conf_mat,  display_labels=classifier.classes_)
    disp.plot(cmap="Oranges", values_format='', xticks_rotation=90)  
    plt.show()
        
    acc.append(np.mean(acc_tmp))
print('najbolja tacnost je u iteraciji broj: ', np.argmax(acc))
tacnost_po_klasi(fin_conf_mat, y_train.unique())
osetljivost_po_klasi(fin_conf_mat, y_train.unique())
preciznost_po_klasi(fin_conf_mat, y_train.unique())
#%%
classifier = KNeighborsClassifier(n_neighbors=10, metric='jaccard')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
#print(conf_mat)

disp = ConfusionMatrixDisplay(confusion_matrix =conf_mat,  display_labels=classifier.classes_)
disp.plot(cmap="Reds", values_format='', xticks_rotation=90)  
plt.show()

print('procenat pogodjenih uzoraka: ', accuracy_score(y_test, y_pred))
print('preciznost mikro: ', precision_score(y_test, y_pred, average='micro'))
print('preciznost makro: ', precision_score(y_test, y_pred, average='macro'))
print('osetljivost mikro: ', recall_score(y_test, y_pred, average='micro'))
print('osetljivost makro: ', recall_score(y_test, y_pred, average='macro'))
print('f mera mikro: ', f1_score(y_test, y_pred, average='micro'))
print('f mera makro: ', f1_score(y_test, y_pred, average='macro'))
print(y_train.unique())
tacnost_po_klasi(fin_conf_mat, y_train.unique())
osetljivost_po_klasi(fin_conf_mat, y_train.unique())
preciznost_po_klasi(fin_conf_mat, y_train.unique())
#%%                         SVM KLASIFIKATOR
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
acc = []
for c in [1, 10, 100]:
    for F in ['linear', 'rbf']:
        for mc in ['ovo', 'ovr']:
            indexes = kf.split(X_train, y_train)
            acc_tmp = []
            fin_conf_mat = np.zeros((len(np.unique(y)),len(np.unique(y_train))))
            for train_index, test_index in indexes:
                classifier = SVC(C=c, kernel=F, decision_function_shape=mc)
                classifier.fit(X_train.iloc[train_index,:], y_train.iloc[train_index])
                y_pred = classifier.predict(X_train.iloc[test_index,:])
                acc_tmp.append(accuracy_score(y_train.iloc[test_index], y_pred))
                fin_conf_mat += confusion_matrix(y.iloc[test_index], y_pred, labels=classifier.classes_)
            print('za parametre C=', c, ', kernel=', F, ' i pristup ', mc, ' tacnost je: ', np.mean(acc_tmp),
                  ' a mat. konf. je:')
            #print(fin_conf_mat)

            disp = ConfusionMatrixDisplay(confusion_matrix =fin_conf_mat,  display_labels=classifier.classes_)
            disp.plot(cmap="Greens", values_format='', xticks_rotation=90)  
            plt.show()

            acc.append(np.mean(acc_tmp))
print('najbolja tacnost je u iteraciji broj: ', np.argmax(acc))
tacnost_po_klasi(fin_conf_mat, y_train.unique())
osetljivost_po_klasi(fin_conf_mat, y_train.unique())
preciznost_po_klasi(fin_conf_mat, y_train.unique())
#%%
classifier = SVC(C=1, kernel='rbf', decision_function_shape='ovr')
classifier.fit(X, y)
y_pred = classifier.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred, labels=classifier.classes_)

#print(conf_mat)
disp = ConfusionMatrixDisplay(confusion_matrix =conf_mat,  display_labels=classifier.classes_)
disp.plot(cmap="Purples", values_format='', xticks_rotation=90)  
plt.show()

print('procenat pogodjenih uzoraka: ', accuracy_score(y_test, y_pred))
print('preciznost mikro: ', precision_score(y_test, y_pred, average='micro'))
print('preciznost makro: ', precision_score(y_test, y_pred, average='macro'))
print('osetljivost mikro: ', recall_score(y_test, y_pred, average='micro'))
print('osetljivost makro: ', recall_score(y_test, y_pred, average='macro'))
print('f mera mikro: ', f1_score(y_test, y_pred, average='micro'))
print('f mera makro: ', f1_score(y_test, y_pred, average='macro'))
tacnost_po_klasi(conf_mat, y_train.unique())
osetljivost_po_klasi(conf_mat, y_train.unique())
preciznost_po_klasi(conf_mat, y_train.unique())