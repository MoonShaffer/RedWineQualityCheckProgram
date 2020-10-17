# RedWineQualityCheckProgram

ワインの成分データを分析して、品質を判別するプログラム
赤ワインデータをダウンロード
In [3]:
from urllib.request import urlretrieve
url = "https://archive.ics.uci.edu" + \
      "/ml/machine-learning-databases/wine-quality" + \
      "/winequality-red.csv"
savepath = "winequality-red.csv"
urlretrieve(url, savepath)
Out[3]:
('winequality-red.csv', <http.client.HTTPMessage at 0x7fbbe6452c88>)
ダウンロードしたデータを読み込む
In [4]:
import pandas as pd
df = pd.read_csv("winequality-red.csv", sep=";", encoding="utf-8")
df
Out[4]:
fixed acidity	volatile acidity	citric acid	residual sugar	chlorides	free sulfur dioxide	total sulfur dioxide	density	pH	sulphates	alcohol	quality
0	7.4	0.700	0.00	1.9	0.076	11.0	34.0	0.99780	3.51	0.56	9.4	5
1	7.8	0.880	0.00	2.6	0.098	25.0	67.0	0.99680	3.20	0.68	9.8	5
2	7.8	0.760	0.04	2.3	0.092	15.0	54.0	0.99700	3.26	0.65	9.8	5
3	11.2	0.280	0.56	1.9	0.075	17.0	60.0	0.99800	3.16	0.58	9.8	6
4	7.4	0.700	0.00	1.9	0.076	11.0	34.0	0.99780	3.51	0.56	9.4	5
5	7.4	0.660	0.00	1.8	0.075	13.0	40.0	0.99780	3.51	0.56	9.4	5
6	7.9	0.600	0.06	1.6	0.069	15.0	59.0	0.99640	3.30	0.46	9.4	5
7	7.3	0.650	0.00	1.2	0.065	15.0	21.0	0.99460	3.39	0.47	10.0	7
8	7.8	0.580	0.02	2.0	0.073	9.0	18.0	0.99680	3.36	0.57	9.5	7
9	7.5	0.500	0.36	6.1	0.071	17.0	102.0	0.99780	3.35	0.80	10.5	5
10	6.7	0.580	0.08	1.8	0.097	15.0	65.0	0.99590	3.28	0.54	9.2	5
11	7.5	0.500	0.36	6.1	0.071	17.0	102.0	0.99780	3.35	0.80	10.5	5
12	5.6	0.615	0.00	1.6	0.089	16.0	59.0	0.99430	3.58	0.52	9.9	5
13	7.8	0.610	0.29	1.6	0.114	9.0	29.0	0.99740	3.26	1.56	9.1	5
14	8.9	0.620	0.18	3.8	0.176	52.0	145.0	0.99860	3.16	0.88	9.2	5
15	8.9	0.620	0.19	3.9	0.170	51.0	148.0	0.99860	3.17	0.93	9.2	5
16	8.5	0.280	0.56	1.8	0.092	35.0	103.0	0.99690	3.30	0.75	10.5	7
17	8.1	0.560	0.28	1.7	0.368	16.0	56.0	0.99680	3.11	1.28	9.3	5
18	7.4	0.590	0.08	4.4	0.086	6.0	29.0	0.99740	3.38	0.50	9.0	4
19	7.9	0.320	0.51	1.8	0.341	17.0	56.0	0.99690	3.04	1.08	9.2	6
20	8.9	0.220	0.48	1.8	0.077	29.0	60.0	0.99680	3.39	0.53	9.4	6
21	7.6	0.390	0.31	2.3	0.082	23.0	71.0	0.99820	3.52	0.65	9.7	5
22	7.9	0.430	0.21	1.6	0.106	10.0	37.0	0.99660	3.17	0.91	9.5	5
23	8.5	0.490	0.11	2.3	0.084	9.0	67.0	0.99680	3.17	0.53	9.4	5
24	6.9	0.400	0.14	2.4	0.085	21.0	40.0	0.99680	3.43	0.63	9.7	6
25	6.3	0.390	0.16	1.4	0.080	11.0	23.0	0.99550	3.34	0.56	9.3	5
26	7.6	0.410	0.24	1.8	0.080	4.0	11.0	0.99620	3.28	0.59	9.5	5
27	7.9	0.430	0.21	1.6	0.106	10.0	37.0	0.99660	3.17	0.91	9.5	5
28	7.1	0.710	0.00	1.9	0.080	14.0	35.0	0.99720	3.47	0.55	9.4	5
29	7.8	0.645	0.00	2.0	0.082	8.0	16.0	0.99640	3.38	0.59	9.8	6
...	...	...	...	...	...	...	...	...	...	...	...	...
1569	6.2	0.510	0.14	1.9	0.056	15.0	34.0	0.99396	3.48	0.57	11.5	6
1570	6.4	0.360	0.53	2.2	0.230	19.0	35.0	0.99340	3.37	0.93	12.4	6
1571	6.4	0.380	0.14	2.2	0.038	15.0	25.0	0.99514	3.44	0.65	11.1	6
1572	7.3	0.690	0.32	2.2	0.069	35.0	104.0	0.99632	3.33	0.51	9.5	5
1573	6.0	0.580	0.20	2.4	0.075	15.0	50.0	0.99467	3.58	0.67	12.5	6
1574	5.6	0.310	0.78	13.9	0.074	23.0	92.0	0.99677	3.39	0.48	10.5	6
1575	7.5	0.520	0.40	2.2	0.060	12.0	20.0	0.99474	3.26	0.64	11.8	6
1576	8.0	0.300	0.63	1.6	0.081	16.0	29.0	0.99588	3.30	0.78	10.8	6
1577	6.2	0.700	0.15	5.1	0.076	13.0	27.0	0.99622	3.54	0.60	11.9	6
1578	6.8	0.670	0.15	1.8	0.118	13.0	20.0	0.99540	3.42	0.67	11.3	6
1579	6.2	0.560	0.09	1.7	0.053	24.0	32.0	0.99402	3.54	0.60	11.3	5
1580	7.4	0.350	0.33	2.4	0.068	9.0	26.0	0.99470	3.36	0.60	11.9	6
1581	6.2	0.560	0.09	1.7	0.053	24.0	32.0	0.99402	3.54	0.60	11.3	5
1582	6.1	0.715	0.10	2.6	0.053	13.0	27.0	0.99362	3.57	0.50	11.9	5
1583	6.2	0.460	0.29	2.1	0.074	32.0	98.0	0.99578	3.33	0.62	9.8	5
1584	6.7	0.320	0.44	2.4	0.061	24.0	34.0	0.99484	3.29	0.80	11.6	7
1585	7.2	0.390	0.44	2.6	0.066	22.0	48.0	0.99494	3.30	0.84	11.5	6
1586	7.5	0.310	0.41	2.4	0.065	34.0	60.0	0.99492	3.34	0.85	11.4	6
1587	5.8	0.610	0.11	1.8	0.066	18.0	28.0	0.99483	3.55	0.66	10.9	6
1588	7.2	0.660	0.33	2.5	0.068	34.0	102.0	0.99414	3.27	0.78	12.8	6
1589	6.6	0.725	0.20	7.8	0.073	29.0	79.0	0.99770	3.29	0.54	9.2	5
1590	6.3	0.550	0.15	1.8	0.077	26.0	35.0	0.99314	3.32	0.82	11.6	6
1591	5.4	0.740	0.09	1.7	0.089	16.0	26.0	0.99402	3.67	0.56	11.6	6
1592	6.3	0.510	0.13	2.3	0.076	29.0	40.0	0.99574	3.42	0.75	11.0	6
1593	6.8	0.620	0.08	1.9	0.068	28.0	38.0	0.99651	3.42	0.82	9.5	6
1594	6.2	0.600	0.08	2.0	0.090	32.0	44.0	0.99490	3.45	0.58	10.5	5
1595	5.9	0.550	0.10	2.2	0.062	39.0	51.0	0.99512	3.52	0.76	11.2	6
1596	6.3	0.510	0.13	2.3	0.076	29.0	40.0	0.99574	3.42	0.75	11.0	6
1597	5.9	0.645	0.12	2.0	0.075	32.0	44.0	0.99547	3.57	0.71	10.2	5
1598	6.0	0.310	0.47	3.6	0.067	18.0	42.0	0.99549	3.39	0.66	11.0	6
1599 rows × 12 columns
データを学習用とテスト用に分割し、モデルを訓練・評価する
利用するアルゴリズム：RandomForest
In [5]:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

wine = pd.read_csv("winequality-red.csv", sep=";", encoding="utf-8")

y = wine["quality"]
x = wine.drop("quality", axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("正解率=", accuracy_score(y_test, y_pred))
             precision    recall  f1-score   support

          3       0.00      0.00      0.00         3
          4       0.50      0.08      0.14        12
          5       0.75      0.81      0.78       144
          6       0.65      0.69      0.67       124
          7       0.66      0.54      0.59        35
          8       0.00      0.00      0.00         2

avg / total       0.68      0.69      0.68       320

正解率= 0.69375
/Users/moonabington/.pyenv/versions/anaconda3-5.3.0/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:29: DeprecationWarning: numpy.core.umath_tests is an internal NumPy module and should not be imported. It will be removed in a future NumPy release.
  from numpy.core.umath_tests import inner1d
/Users/moonabington/.pyenv/versions/anaconda3-5.3.0/lib/python3.7/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
正解率の精度向上を行う
In [7]:
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd

wine = pd.read_csv("winequality-red.csv", sep=";", encoding="utf-8")

count_data = wine.groupby('quality')["quality"].count()
print(count_data)

count_data.plot()
plt.savefig("wine-count-plt.png")
plt.show()
quality
3     10
4     53
5    681
6    638
7    199
8     18
Name: quality, dtype: int64

In [10]:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

wine = pd.read_csv("winequality-red.csv", sep=";", encoding="utf-8")

y = wine["quality"]
x = wine.drop("quality", axis=1)

newlist = []
for v in list(y):
    if v <= 4:
        newlist += [0]
        
    elif v <= 7:
        newlist += [1]
    
    else:
        newlist += [2]
        
y = newlist

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("正解率=", accuracy_score(y_test, y_pred))
             precision    recall  f1-score   support

          0       0.29      0.25      0.27         8
          1       0.97      0.98      0.98       310
          2       0.00      0.00      0.00         2

avg / total       0.95      0.96      0.95       320

正解率= 0.95625
正解率が向上しました
