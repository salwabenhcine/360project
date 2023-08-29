from django.template.context_processors import request
import mysql.connector as Mc
# Load the necessary python libraries
import pandas as pd
from datetime import date

from rest_framework.response import Response
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from rest_framework.decorators import api_view


conn = Mc.connect(host="127.0.0.1",
                      user="root", password="",
                      database="site",
                      )
cursor = conn.cursor()
req = 'SELECT date, id_client , sexe  , situation ,dateE1, dateE2, dateE3, dateE4, profession, gouvernorat,' \
          'marque1, marque2, marque3, marque4,marque5 FROM nvclients '
cursor.execute(req)
clients = cursor.fetchall()

cursor = conn.cursor()
req2 = 'SELECT * FROM nvachats'
cursor.execute(req2)
achats = cursor.fetchall()

df = pd.DataFrame(clients)
df.columns = ['age', 'id', 'sexe', 'situation', 'dateE1', 'dateE2', 'dateE3', 'dateE4', 'profession', 'region',
                      'marque1', 'marque2', 'marque3', 'marque4', 'marque5']
df1 = pd.DataFrame(achats)
df1.columns = ['date', 'comm', 'id', 'cat', 'souscat', 'article', 'code', 'codeb', 'prix', 'quant', 'marque']
df1.to_csv('C:/Users/slwa/Downloads/my_data.csv',sep=',')
data=pd.read_csv('C:/Users/slwa/Downloads/my_data.csv',index_col='date',parse_dates=True)
df = df.replace('ben Arous', 'ben arous')  # replace 'ben Arous' with 'ben arous'
cursor = conn.cursor()
req = 'SELECT CODE , article  , Prix_de_vente ,marque  FROM produitss'
cursor.execute(req)
produits = cursor.fetchall()
dfp =pd.DataFrame(produits)
dfp.columns = ['code','article', 'prix' ,'marque']

def from_dob_to_age(born):
    if born is not None:
        today = date.today()
        return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


df['dateE1'] = pd.to_datetime(df['dateE1'], format="%Y-%m-%d")  # Convert  df['dateE1'] to Datetime
df['dateE1'] = df['dateE1'].apply(from_dob_to_age)  # Apply the function on the 'dateE1' column
df['dateE2'] = pd.to_datetime(df['dateE2'], format="%Y-%m-%d")  # Convert  df['dateE2'] to Datetime
df['dateE2'] = df['dateE2'].apply(from_dob_to_age)  # Apply the function on the 'dateE2' column
df['dateE3'] = pd.to_datetime(df['dateE3'], format="%Y-%m-%d")  # Convert  df['dateE3'] to Datetime
df['dateE3'] = df['dateE3'].apply(from_dob_to_age)  # Apply the function on the 'dateE3' column
df['dateE4'] = pd.to_datetime(df['dateE4'], format="%Y-%m-%d")  # Convert  df['dateE4'] to Datetime
df['dateE4'] = df['dateE4'].apply(from_dob_to_age)  # Apply the function on the 'dateE4' column
df['age'] = pd.to_datetime(df['age'], format="%Y-%m-%d")
df['age'] = df['age'].apply(from_dob_to_age)  # Apply the function on the 'age' column
df.loc[(df['dateE1'] >= 0) & (df['dateE1'] <= 1), 'dateE1'] = 1
df.loc[(df['dateE1'] > 1) & (df['dateE1'] <= 3), 'dateE1'] = 2
df.loc[(df['dateE1'] > 3) & (df['dateE1'] <= 5), 'dateE1'] = 3
df.loc[(df['dateE1'] > 5) & (df['dateE1'] <= 12), 'dateE1'] = 4
df.loc[(df['dateE1'] > 12), 'dateE1'] = 5
df.loc[(df['dateE2'] >= 0) & (df['dateE2'] <= 1), 'dateE2'] = 1
df.loc[(df['dateE2'] > 1) & (df['dateE2'] <= 3), 'dateE2'] = 2
df.loc[(df['dateE2'] > 3) & (df['dateE2'] <= 5), 'dateE2'] = 3
df.loc[(df['dateE2'] > 5) & (df['dateE2'] <= 12), 'dateE2'] = 4
df.loc[(df['dateE2'] > 12), 'dateE2'] = 5
df.loc[(df['dateE3'] >= 0) & (df['dateE3'] <= 1), 'dateE3'] = 1
df.loc[(df['dateE3'] > 1) & (df['dateE3'] <= 3), 'dateE3'] = 2
df.loc[(df['dateE3'] > 3) & (df['dateE3'] <= 5), 'dateE3'] = 3
df.loc[(df['dateE3'] > 5) & (df['dateE3'] <= 12), 'dateE3'] = 4
df.loc[(df['dateE3'] > 12), 'dateE3'] = 5
df.loc[(df['dateE4'] >= 0) & (df['dateE4'] <= 1), 'dateE4'] = 1
df.loc[(df['dateE4'] > 1) & (df['dateE4'] <= 3), 'dateE4'] = 2
df.loc[(df['dateE4'] > 3) & (df['dateE4'] <= 5), 'dateE4'] = 3
df.loc[(df['dateE4'] > 5) & (df['dateE4'] <= 12), 'dateE4'] = 4
df.loc[(df['dateE4'] > 12), 'dateE4'] = 5
labelencoder = LabelEncoder()
df['sexe'] = labelencoder.fit_transform(df['sexe'])
df['situation'] = labelencoder.fit_transform(
    df['situation'])  # view the values of the 'situation' columns before preprocessing
df['profession'] = labelencoder.fit_transform(
    df['region'])  # view the values of the 'region' columns before preproces sing
df['region'] = labelencoder.fit_transform(df['region'])
X = pd.DataFrame(df, columns=['sexe', 'region', 'age', 'situation'])
X['age'] = labelencoder.fit_transform(X['age'])  # Fit label encoder and return encoded labels
X['situation'] = labelencoder.fit_transform(X['situation'])
X['region'] = labelencoder.fit_transform(X['region'])


def purchases(df):  # This function of customer purchases.
    a = {}
    for i in range(len(df)):
        if df.id[i] in a:
            if df.marque[i] in a[df.id[i]]:
                if df.code[i] in a[df.id[i]][df.marque[i]]:
                    a[df.id[i]][df.marque[i]][df.code[i]] += df.quant[i]
                else:
                    a[df.id[i]][df.marque[i]][df.code[i]] = df.quant[i]
            else:
                a[df.id[i]].update({df.marque[i]: {}})
                a[df.id[i]][df.marque[i]].update({df.code[i]: df.quant[i]})
        else:
            a[df.id[i]] = {}
            a[df.id[i]].update({df.marque[i]: {}})
            a[df.id[i]][df.marque[i]].update({df.code[i]: df.quant[i]})
    return a


a = purchases(df1)
@api_view(["POST"])
def Produits (request) :
    id = request.data["id"]
    n_neighbors = request.data["n_neighbors"]
    datepd = request.data["datepd"]
    datepf = request.data["datepf"]
    datecd = request.data["datecd"]
    datecf = request.data["datecf"]

    def achatre (a, id):
        dr = dict()
        dac = dict()
        d = dict()
        for k, v in a.items():
            if id == k:
                dr[id] = v
            for kv, vv in dr.items():
                for kk, vk in vv.items():
                    dac[kk] = vk
                    for m, n in dac.items():
                        for km, vn in n.items():
                            d[km] = vn
        l = sorted(d.items(), key=lambda t: t[1], reverse=True)
        return l

    def similaires(X, id, n_neighbors):
        i = (df[df['id'] == id].index)
        n = n_neighbors
        l = []
        XX = pd.DataFrame(X, columns=['sexe', 'region', 'age', 'situation'])
        neigh = NearestNeighbors(n_neighbors=n)
        neigh.fit(XX)
        Z = neigh.kneighbors([XX.to_numpy()[i]][0])
        l = list(Z[1][0])
        if i in l:
            l.pop(l.index(i))
            k = list(df['id'].iloc[l])
        return k

    def similar_purchases(df, X, ID, n_neighbors, a):
        dict_purchases = {}
        for IDj in similaires(X, ID, n_neighbors):
            if IDj not in a:
                continue
            else:
                for k, v in a[IDj].items():
                    if k not in dict_purchases:
                        dict_purchases[k] = v
                    else:
                        for kk, vv in v.items():
                            if kk not in dict_purchases[k]:
                                dict_purchases[k][kk] = vv
                            else:
                                dict_purchases[k][kk] += vv
        if ID in a:
            if dict_purchases != {}:
                for k, v in a[ID].items():
                    if k not in dict_purchases:
                        dict_purchases[k] = v
                    else:
                        for kk, vv in v.items():
                            if kk not in dict_purchases[k]:
                                dict_purchases[k][kk] = vv
                            else:
                                dict_purchases[k][kk] += vv
            else:
                dict_purchases = a[ID]

        a[ID] = dict_purchases
        return dict_purchases

    mydict = dict(zip(df1['id'], zip(df['marque1'], df['marque2'], df['marque3'], df['marque4'], df['marque5'])))

    def purchases_sep(d, mydict):
        ass = {}
        for K, V in d.items():
            ass[K] = [{}, {}]
            for k, v in V.items():
                if K in mydict:
                    if k in mydict[K]:
                        ass[K][0].update({k: v})
                    else:
                        ass[K][1].update({k: v})
        return ass

    ass = purchases_sep(a, mydict)

    def top_purchases_products(a):
        d = {}  # create an empty dictionary
        for V in a.values():
            for k, v in V.items():
                for kk, vv in v.items():
                    if kk not in d:
                        d[(k, kk)] = vv
                    else:
                        d[(k, kk)] += vv
        l = sorted(d.items(), key=lambda t: t[1], reverse=True)
        return l

    def top_purchases_similar_products(a, X, id, n_neighbors):
        ds = {}  # create an empty dictionary
        ls = similaires(X, id, n_neighbors)
        for j in ls:
            if j in a:
                ds.update({j: a[j]})
        k = top_purchases_products(ds)
        return k

    df2 = pd.read_csv('C:/Users/slwa/Downloads/my_data.csv', index_col='date', parse_dates=True)
    df_previous = df2.loc[datepd:datepf]  # display a part of the df in a time interval
    df_current = df2.loc[datecd:datecf]
    a_current = purchases(df_current)
    a_previous = purchases(df_previous)

    def trends_purchases_products(current, previous):
        d_current = {}  # create an empty dictionary
        d = {}
        l = []
        for V in a_current.values():
            for k, v in V.items():
                for kv, vv in v.items():
                    if kv not in d:
                        d_current[kv] = sum(v.values())
                    else:
                        d_current[kv] += sum(v.values())
        d_previous = {}
        for V in a_previous.values():
            for k, v in V.items():
                for kv, vv in v.items():
                    if k not in d:
                        d_previous[kv] = sum(v.values())  # sum of occurrence numbers
                    else:
                        d_previous[kv] += sum(v.values())
        for k in d_current:
            if k in d_previous:
                d[k] = (d_current[k] - d_previous[k]) / d_previous[k]
                # calculate the average of product purchases
        for k, v in d.items():
            if v > 0:
                l.append(k)

        return l


    def top_purchases_products_preferred(ass):
        dp = {}
        for i in ass.keys():
            dp[i] = ass[i][0]
        topp = top_purchases_products(dp)
        return topp

    def top_purchases_similer_products_preferred(ass, X, id, n_neighbors):
        db = {}
        for ii in ass.keys():
            db[ii] = ass[ii][0]
        tops = top_purchases_similar_products(db, X, id, n_neighbors)
        return (tops)

    def affichageproduits(a_current, a_previous, a, ass, X, id, n_neighbors):
        mySet = set()
        if id in purchases(df1).keys():
            lr = achatre(a, id)
            lre = lr[:5]
            for i in range(len(lre)):
                mySet.add(lre[i][0])
            l = top_purchases_products(a)
            m = l[:5]
            for i in range(len(m)):
                mySet.add(m[i][0][1])
            ls = top_purchases_similar_products(a, X, id, n_neighbors)
            lss = ls[:5]
            for i in range(len(lss)):
                mySet.add(lss[i][0][1])
            lp = top_purchases_products_preferred(ass)
            lpp = lp[:5]
            for i in range(len(lpp)):
                mySet.add(lpp[i][0][1])
            t = trends_purchases_products(a_current, a_previous)
            for i in t:
                mySet.add(i)
            ln = len(mySet)
            if ln < 26:
                ls = top_purchases_similar_products(a, X, id, n_neighbors)
                lss = ls[5:]
                for i in range(len(lss)):
                    mySet.add(lss[i][0][1])
                    ln = ln + 1
                    if (ln < 26):
                        continue
                    else:
                        break
        else:
            l = top_purchases_products(a)
            m = l[:10]
            for i in range(len(m)):
                mySet.add(m[i][0][1])
            ls = top_purchases_similar_products(a, X, id, n_neighbors)
            lss = ls[:10]
            for i in range(len(lss)):
                mySet.add(lss[i][0][1])
            t = trends_purchases_products(a_current, a_previous)
            for i in t:
                mySet.add(i)
            ln = len(mySet)
            if ln < 25:
                ls = top_purchases_similar_products(a, X, id, n_neighbors)
                lss = ls[10:]
                for i in range(len(lss)):
                    mySet.add(lss[i][0][1])
                    ln = ln + 1
                    if (ln < 25):
                        continue
                    else:
                        break

        return (mySet)

    m = affichageproduits(a_current, a_previous, a, ass, X, id,n_neighbors)
    l = set()
    for i in m:
        for j in range(len(df1)):
            if df1.code[j] == i:
                l.add(df1.article[j])

    return Response({id: l,"code":m})




@api_view(["POST"])
def Marque( request):
    id = request.data["id"]
    n_neighbors = request.data["n_neighbors"]
    datepd = request.data["datepd"]
    datepf = request.data["datepf"]
    datecd = request.data["datecd"]
    datecf = request.data["datecf"]
    dateD= request.data["dateD"]
    dateF = request.data["dateF"]
    d1min = request.data["d1min"]
    d1max = request.data["d1max"]
    d2min = request.data["d2min"]
    d2max = request.data["d2max"]
    d3min = request.data["d3min"]
    d3max = request.data["d3max"]
    d4min = request.data["d4min"]
    d4max = request.data["d4max"]

    def achatmarque(a, id):
        dr = dict()
        dac = dict()
        for k, v in a.items():
            if id == k:
                dr[id] = v
            for kv, vv in dr.items():
                for kk, vk in vv.items():
                        dac[kk] = sum(vk.values())
            l = sorted(dac.items(), key=lambda t: t[1], reverse=True)
            return (l)
            
    def similaires(X, id, n_neighbors):
        i = (df[df['id'] == id].index)
        n = n_neighbors
        l = []
        XX = pd.DataFrame(X, columns=['sexe', 'region', 'age', 'situation'])
        neigh = NearestNeighbors(n_neighbors=n)
        neigh.fit(XX)
        Z = neigh.kneighbors([XX.to_numpy()[i]][0])
        l = list(Z[1][0])
        if i in l:
            l.pop(l.index(i))
            k = list(df['id'].iloc[l])
        return k
    mydict = dict(zip(df1['id'], zip(df['marque1'], df['marque2'], df['marque3'], df['marque4'], df['marque5'])))

    def purchases_sep(d, mydict):
        ass = {}
        for K, V in d.items():
            ass[K] = [{}, {}]
            for k, v in V.items():
                if K in mydict:
                    if k in mydict[K]:
                        ass[K][0].update({k: v})
                    else:
                        ass[K][1].update({k: v})
        return ass

    ass = purchases_sep(a, mydict)

    def top_purchases_brands(a):
        d = {}  # create an empty dictionary
        for V in a.values():
            for k, v in V.items():
                if k not in d:
                    d[k] = sum(v.values())
                else:
                    d[k] += sum(v.values())
        l = sorted(d.items(), key=lambda t: t[1], reverse=True)  # sort dictionary elements in decreasing order.
        return l

    l = top_purchases_brands(a)

    def top_purchases_similar_brands(a, X, id, n_neighbors):
        ds = {}  # create an empty dictionary
        ls = similaires(X, id, n_neighbors)
        for j in ls:
            if j in a:
                ds.update({j: a[j]})
        k = top_purchases_brands(ds)
        return k


    def trends_purchases_brands(current, previous):
        d_current = {}  # create an empty dictionary
        d = {}
        l = []
        for V in current.values():
            for k, v in V.items():
                if k not in d:
                    d_current[k] = sum(v.values())
                else:
                    d_current[k] += sum(v.values())
        d_previous = {}
        for V in previous.values():
            for k, v in V.items():
                if k not in d:
                    d_previous[k] = sum(v.values())  # sum of occurrence numbers
                else:
                    d_previous[k] += sum(v.values())
        for k in d_current:
            if k in d_previous:
                d[k] = (d_current[k] - d_previous[k]) / d_previous[k]
                # calculate the average of product purchases
        for k, v in d.items():
            if v > 0:
                l.append(k)

        return l
    df2 = pd.read_csv('C:/Users/slwa/Downloads/my_data.csv', index_col='date', parse_dates=True)
    df_previous = df2.loc[datepd:datepf]  # display a part of the df in a time interval
    df_current = df2.loc[datecd:datecf]
    a_current = purchases(df_current)
    a_previous = purchases(df_previous)
    def top_purchases_brands_preferred(ass):
        db = {}
        for i in ass.keys():
            db[i] = ass[i][0]
        topb = top_purchases_brands(db)
        return topb
    def top_purchases_similer_brands_preferred (df, ass, X, id, n_neighbors):
        db = {}
        for ii in ass.keys():
            db[ii] = ass[ii][0]
        topbs = top_purchases_similar_brands(db, X, id, n_neighbors)
        return (topbs)

    def somme ( id , dateD, dateF):
        df4 = pd.read_csv('C:/Users/slwa/Downloads/my_data.csv', index_col='date', parse_dates=True)
        df3 = df4.loc[dateD:dateF]
        ap = purchases(df3)
        l = list()
        dp = dict()
        for k, v in ap.items():
            if k == id:
                l = list(ap[id].items())
        for i in range(len(l)):
            m = (l[i][1])
            ls = []
            for c, b in m.items():
                for j in range(len(dfp)):
                    if dfp.code[j] == c:
                        pr = dfp.prix[j]
                        pr = int(pr)
                        p = pr * b
                        ls.append(p)
                        dp[l[i][0]] = sum(ls)
        lp = list(dp.items())
        return (lp)
    c = somme ( id , dateD, dateF)
    print(c)
    def affichagemarques(a_current, a_previous, a, ass, X, id, n_neighbors, dateD, dateF, d1min, d1max, d2min, d2max, d3min, d3max, d4min, d4max):
        mySet = set(dict())
        d = dict()
        if id in purchases(df1).keys():
            lr = somme(id, dateD, dateF)
            lre = lr[:5]
            for i in range(len(lre)):
                if lre[i][1] in range(d1min, d1max):
                    d[lre[i][0]] = 'deal1'
                    mySet.update(d.items())
                elif lre[i][1] in range(d2min, d2max):
                    d[lre[i][0]] = 'deal2'
                    mySet.update(d.items())
                elif lre[i][1] in range(d3min, d3max):
                    d[lre[i][0]] = 'deal3'
                    mySet.update(d.items())
                elif lre[i][1] in range(d4min, d4max):
                    d[lre[i][0]] = 'deal4'
                    mySet.update(d.items())
                else:
                    d[lre[i][0]] = 'deal5'
                    mySet.update(d.items())
            print(d)
            l = top_purchases_brands(a)
            m = l[:5]
            for i in range(len(m)):
                if m[i][0] not in d.keys():
                    d.update({m[i][0]: 'deal1'})
            ls = top_purchases_similar_brands(a, X, id, n_neighbors)
            lss = ls[:5]
            for i in range(len(lss)):
                if lss[i][0] not in d.keys():
                    d.update({lss[i][0]: 'deal1'})
            lp = top_purchases_brands_preferred(ass)
            lpp = lp[:5]
            for i in range(len(lpp)):
                if lpp[i][0] not in d.keys():
                    d.update({lpp[i][0]: 'deal1'})
            t = trends_purchases_brands(a_current, a_previous)
            for i in t:
                if i not in d.keys():
                    d.update({i: 'deal1'})
            ln = len(d)
            print(mySet)
            print(ln)
            while ln < 20:
                ls = top_purchases_similar_brands(a, X, id, n_neighbors)
                lss = ls[5:]
                for i in range(len(lss)):
                    if lss[i][0] not in d.keys():
                        d.update({lss[i][0]: 'deal1'})
                        ln = ln + 1
                    if (ln < 20):
                        continue
                    else:
                        break
        else:
            l = top_purchases_brands(a)
            m = l[:10]
            for i in range(len(m)):
                d.update({m[i][0]: 'deal1'})
            ls = top_purchases_similar_brands(a, X, id, n_neighbors)
            lss = ls[:10]
            for i in range(len(lss)):
                d.update({lss[i][0]: 'deal1'})
            t = trends_purchases_brands(a_current, a_previous)
            for i in t:
                d.update({i: 'deal1'})
            ln = len(d)
            while ln < 20:
                ls = top_purchases_similar_brands(a, X, id, n_neighbors)
                lss = ls[10:]
                for i in range(len(lss)):
                    d.update({lss[i][0]: 'deal1'})
                    ln = ln + 1
                    if ln < 20:
                        continue
                    else:
                        break
        mySet.update(d.items())
        return mySet

    marque = affichagemarques(a_current, a_previous, a, ass, X, id, n_neighbors, dateD, dateF, d1min, d1max, d2min, d2max, d3min, d3max, d4min, d4max)
    print(marque)
    return Response({id: marque})

