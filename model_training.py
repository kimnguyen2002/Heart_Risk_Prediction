import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import joblib #Import joblib here


# Đọc dữ liệu đã làm sạch
df = pd.read_csv('heart.csv')

# Chuẩn bị dữ liệu
X = df.drop('target', axis=1)
y = df['target']

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler.  This is crucial!
joblib.dump(scaler, 'scaler.pkl')


# Huấn luyện mô hình Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)
log_reg_pred = log_reg.predict(X_test_scaled)
log_reg_acc = accuracy_score(y_test, log_reg_pred)

# Huấn luyện mô hình Random Forest
rf = RandomForestClassifier()
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, rf_pred)

# Huấn luyện mô hình SVM
svm = SVC()
svm.fit(X_train_scaled, y_train)
svm_pred = svm.predict(X_test_scaled)
svm_acc = accuracy_score(y_test, svm_pred)

# So sánh độ chính xác của các mô hình
print(f"Logistic Regression Accuracy: {log_reg_acc}")
print(f"Random Forest Accuracy: {rf_acc}")
print(f"SVM Accuracy: {svm_acc}")

# Chọn mô hình tốt nhất dựa trên độ chính xác
best_model = None
if log_reg_acc > rf_acc and log_reg_acc > svm_acc:
    best_model = log_reg
    print("Best Model: Logistic Regression")
elif rf_acc > log_reg_acc and rf_acc > svm_acc:
    best_model = rf
    print("Best Model: Random Forest")
else:
    best_model = svm
    print("Best Model: SVM")

# Lưu mô hình tốt nhất
joblib.dump(best_model, 'best_model.pkl')