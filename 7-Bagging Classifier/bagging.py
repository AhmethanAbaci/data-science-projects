# Gerekli kütüphanelerin import edilmesi
from sklearn.ensemble import BaggingClassifier   # torbalama modeli
from sklearn.tree import DecisionTreeClassifier  # torbalama icerisinde kullanacagimiz veri seti
from sklearn.datasets import load_iris           # Örnek olarak Iris veri seti
from sklearn.model_selection import train_test_split  # Veri setini train/test olarak ayırmak için fonskiyon
from sklearn.metrics import accuracy_score       # Modelin doğruluk (accuracy) skorunu ölçmek için


# load data set
iris = load_iris()

# data train and test split
X = iris.data       # features
y = iris.target     # target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# defined base model: decision tree
base_model = DecisionTreeClassifier(random_state=42)

# create bagging model
bagging_model = BaggingClassifier(
    estimator=base_model,     # temel model (yeni parametre adı)
    n_estimators=10,          # kullanılacak model sayısı
    max_samples=0.2,          # her modelin kullanacağı örnek oranı
    max_features=0.8,         # her modelin kullanacağı özellik oranı
    bootstrap=True,           # örneklerin tekrar seçilmesine izin ver
    random_state=42,
    n_jobs=-1                 # paralel işlem (tüm CPU çekirdekleri)
)

# model training
bagging_model.fit(X_train, y_train)

# model testing
y_pred = bagging_model.predict(X_test)

# accuracy score
print("Accuracy:", accuracy_score(y_test, y_pred))

