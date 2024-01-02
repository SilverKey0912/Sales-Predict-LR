import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

app = Flask(__name__)

# Đường dẫn đến file model.joblib
model_path = r"C:\Users\lymin\Documents\Semester 7\Big Data Analyst\Web_predict\Code\linear_model.joblib"
linear_model = joblib.load(model_path)

# Route cho trang chủ
@app.route('/')
def index():
    return render_template('index.html')

# Route xử lý dự đoán khi nhận file CSV từ form
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        print("Predict route is called.")
        # Lấy file từ form
        uploaded_file = request.files['file']

        if uploaded_file.filename != '':
            # Đọc file CSV
            data = pd.read_csv(uploaded_file)

            # Chọn những cột cần thiết
            selected_columns = ['QUANTITYORDERED', 'PRICEEACH', 'ORDERLINENUMBER', 'SALES', 'PRODUCTLINE', 'MSRP', 'DEALSIZE']
            new_data = data[selected_columns]

            # Chuẩn hóa các cột không phải kiểu số bằng LabelEncoder
            labelencoder = LabelEncoder()
            for column in new_data.columns:
                if new_data[column].dtype == 'object':
                    new_data[column] = labelencoder.fit_transform(new_data[column])

            # Chuẩn hóa tất cả các cột sử dụng StandardScaler
            scaler = StandardScaler()
            new_data_scaled = scaler.fit_transform(new_data.drop('SALES', axis=1))

            # Dự đoán sử dụng mô hình đã được huấn luyện
            predicted_sales = linear_model.predict(new_data_scaled)

            # Tính mean và std của cột "SALES"
            mean_sales = new_data['SALES'].mean()
            std_sales = new_data['SALES'].std()

            # Áp dụng công thức z-score để chuyển đổi giá trị dự đoán về giá trị gốc
            predicted_sales_original = predicted_sales * std_sales + mean_sales

            # Thêm cột 'PREDICTED_SALES' vào DataFrame mới
            data_with_predictions = pd.concat([data[['ORDERNUMBER']], pd.DataFrame({'PREDICTED_SALES': predicted_sales_original})], axis=1)

            # Chuyển đổi kết quả thành HTML để hiển thị
            result_data = data_with_predictions[['ORDERNUMBER', 'PREDICTED_SALES']].to_dict(orient='records')

            return render_template('index.html', result_data=result_data)

    return 'Invalid file or no file provided.'

if __name__ == '__main__':
    app.run(debug=True)
