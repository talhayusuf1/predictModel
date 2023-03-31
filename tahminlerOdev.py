#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
import statsmodels.api as sm # p value ların hesaplanması
#1.0 veri yukleme
veriler = pd.read_csv('maaslar_yeni.csv')

#1.1 kolon temizleme işlemi yapıldı
x = veriler.iloc[:,2:5]# Bu bağımsız degisken sonuca etkisi vardır
y = veriler.iloc[:,5:]# Bu bagımlı bir degiskendir yani bulmak istedigimiz
X = x.values
Y = y.values


#linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

print("Linear OLS")
model = sm.OLS(lin_reg.predict(X),X)#  X degerinin tahminini al X le karsılastır
print(model.fit().summary())

print('Linear R2 degeri')
print(r2_score(Y, lin_reg.predict(X)))


#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)


print("poly OLS")
model2 = sm.OLS(lin_reg2.predict(poly_reg.fit_transform(X)),X)#  X degerinin tahminini al X le karsılastır
print(model2.fit().summary())

print('Polynomial R2 degeri')
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))


# Decision Tree de olceklemeye ihtiyac duyulkmuyor
#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc1=StandardScaler()

x_olcekli = sc1.fit_transform(X)

sc2=StandardScaler()
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))


from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)

print("svr OLS")
model3 = sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)#  X degerinin tahminini al X le karsılastır
print(model3.fit().summary())


print('Support Vector R2 degeri')
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))

from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state = 0)
r_dt.fit(X,Y)

print("dt OLS")
model4 = sm.OLS(r_dt.predict(X),X)#  X degerinin tahminini al X le karsılastır
print(model4.fit().summary())

print('Decision Tree R2')# BURADA  ALGORİTMAMIZI TEST EDİYORUZ DOGRULUGUNU
print(r2_score(Y, r_dt.predict(X)))

from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
rf_reg.fit(X,Y.ravel())


print("rf OLS")
model5 = sm.OLS(rf_reg.predict(X),X)#  X degerinin tahminini al X le karsılastır
print(model5.fit().summary())

print('Random Forest R2 degeri')
print(r2_score(Y,rf_reg.predict(X)))


# Ozet r2 degerleri
print('-----------------------')

linearModel = r2_score(Y, lin_reg.predict(X))
print(f"Linear R2 degeri:{linearModel}")


polynomialModel = r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X)))
print(f'Polynomial R2 degeri:{polynomialModel}')


svrModel = r2_score(y_olcekli, svr_reg.predict(x_olcekli))
print(f'Support Vector R2 degeri:{svrModel}')


dtModel = r2_score(Y, r_dt.predict(X))
print(f'Decision Tree R2 degeri:{dtModel}')

rfModel = r2_score(Y,rf_reg.predict(X))
print(f'Random Forest R2 degeri:{rfModel}')




















