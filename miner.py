import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import datetime as dt

class DataMinerMixin:
    
    #Thuật toán K-Means: Gom cụm khách hàng
    def cluster_customers(self, n_clusters=3):
        print(f"🧠 Đang chạy K-Means để chia khách hàng thành {n_clusters} nhóm...")
        try:
            if not hasattr(self, 'rfm_df') or self.rfm_df.empty:
                print("❌ Lỗi: Chưa có dữ liệu RFM. Hãy chạy calculate_rfm() trước.")
                return

            scaler = StandardScaler()
            rfm_scaled = scaler.fit_transform(self.rfm_df[['Recency', 'Frequency', 'Monetary']])
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)
            
            print("✅ Đã phân cụm khách hàng thành công!")
            print(self.rfm_df['Cluster'].value_counts())
        except Exception as e:
            print(f"❌ Lỗi khi chạy K-Means: {e}")

    #Thuật toán Apriori: Tìm luật mua chung
    def mine_association_rules(self, min_support=0.5, min_confidence=0.2): # Ép cứng mặc định 0.01 để test
        print(f"🧠 Đang chạy Apriori tìm quy luật giỏ hàng (Support>={min_support}, Conf>={min_confidence})...")
        try:
            if not hasattr(self, 'basket_df') or self.basket_df.empty:
                print("❌ Lỗi: Chưa có dữ liệu giỏ hàng. Hãy chạy prepare_basket_data() trước.")
                return

            so_hoa_don = self.basket_df.shape[0]
            so_san_pham = self.basket_df.shape[1]
            tong_so_true = self.basket_df.sum().sum()
            print(f"🔍 [Kiểm tra]: Ma trận có {so_hoa_don} hóa đơn và {so_san_pham} sản phẩm.")
            print(f"🔍 [Kiểm tra]: Có tổng cộng {tong_so_true} giao dịch hợp lệ (ô True).")
            
            if tong_so_true == 0:
                print("❌ CHẾT DỞ: Giỏ hàng trống trơn (Toàn False). Lỗi nằm ở bước tạo giỏ hàng!")
                return

            # Tìm các tập phổ biến
            frequent_itemsets = apriori(self.basket_df, min_support=min_support, use_colnames=True)
            print(f"🔍 [Vòng 1]: Đã tìm ra {len(frequent_itemsets)} mặt hàng/nhóm mặt hàng đạt chuẩn Support.")
            
            if frequent_itemsets.empty:
                print("⚠️ [Vòng 1 Thất bại]: Không món nào đạt chuẩn. Hãy thử giảm min_support xuống 0.005 hoặc 0.001.")
                self.rules = pd.DataFrame()
                return

            # Sinh ra các luật kết hợp
            self.rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
            
            if self.rules.empty:
                print("⚠️ [Vòng 2 Thất bại]: Có mặt hàng phổ biến, nhưng chúng không bao giờ mua chung với nhau đủ nhiều (Thiếu Confidence).")
                return
                
            # Sắp xếp theo Lift
            self.rules = self.rules.sort_values('lift', ascending=False)
            print(f"✅ THÀNH CÔNG: Đã tìm thấy {len(self.rules)} quy luật mua hàng hấp dẫn!")
            
        except Exception as e:
            print(f"❌ Lỗi khi chạy Apriori: {e}")

    #Thuật toán Hồi quy (Regression): Dự báo doanh thu
    def forecast_sales(self, date_col='InvoiceDate', amount_col='TotalAmount', periods=30):
        print(f"🧠 Đang xây dựng mô hình dự báo doanh thu cho {periods} ngày tới...")
        try:
            # Gom tổng doanh thu theo từng ngày
            daily_sales = self.df.groupby(pd.to_datetime(self.df[date_col]).dt.date)[amount_col].sum().reset_index()
            daily_sales.columns = ['Date', 'Sales']
            daily_sales['Date'] = pd.to_datetime(daily_sales['Date'])
            daily_sales = daily_sales.sort_values('Date')
            
            daily_sales['DayIndex'] = np.arange(len(daily_sales))
            
            # Huấn luyện mô hình Linear Regression
            X = daily_sales[['DayIndex']]
            y = daily_sales['Sales']
            model = LinearRegression()
            model.fit(X, y)
            
            # Dự báo tương lai
            last_day_index = daily_sales['DayIndex'].max()
            future_indices = np.arange(last_day_index + 1, last_day_index + 1 + periods).reshape(-1, 1)
            future_sales = model.predict(pd.DataFrame(future_indices, columns=['DayIndex']))
            
            # Tạo DataFrame kết quả dự báo
            last_date = daily_sales['Date'].max()
            future_dates = [last_date + dt.timedelta(days=i) for i in range(1, periods + 1)]
            
            self.forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted_Sales': future_sales
            })
            
            self.daily_sales_history = daily_sales 
            
            print("✅ Đã hoàn tất dự báo doanh thu tương lai!")
        except Exception as e:
            print(f"❌ Lỗi khi dự báo doanh thu: {e}")

    #Thuật toán phân lớp: trích luật phân loại khách hàng
    def classify_customers(self):
        print("🌳 Đang xây dựng Cây Quyết Định (Decision Tree)...")
        try:
            # Kiểm tra xem đã chạy K-Means trước đó chưa
            if not hasattr(self, 'rfm_df') or 'Cluster' not in self.rfm_df.columns:
                print("❌ Lỗi: Chưa có dữ liệu phân cụm. Vui lòng chạy K-Means trước.")
                return

            X = self.rfm_df[['Recency', 'Frequency', 'Monetary']]
            y = self.rfm_df['Cluster']

            # Khởi tạo mô hình Decision Tree
            dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=42)
            dt_classifier.fit(X, y)

            # Đánh giá độ chính xác
            accuracy = dt_classifier.score(X, y)
            print(f"🎯 Độ chính xác khi dịch K-Means sang Luật if-else: {accuracy*100:.2f}%")

            # Vẽ sơ đồ Cây Quyết Định
            plt.figure(figsize=(16, 10))
            plot_tree(dt_classifier, 
                      feature_names=['Recency (Ngày)', 'Frequency (Lần)', 'Monetary (Tiền)'],  
                      class_names=['Nhóm 0 (Nguy cơ)', 'Nhóm 1 (VIP)', 'Nhóm 2 (Tiềm năng)'], 
                      filled=True, 
                      rounded=True, 
                      fontsize=12)
            
            plt.title(f"Sơ đồ Cây Quyết Định Phân Loại Khách Hàng (Accuracy: {accuracy*100:.2f}%)", fontsize=18, fontweight='bold')
            plt.tight_layout()
            
            # Lưu ảnh
            import os
            os.makedirs('figures', exist_ok=True)
            plt.savefig('figures/decision_tree_rules.png', dpi=300)
            plt.close()
            
            print("✅ Đã lưu sơ đồ Cây Quyết Định tại: figures/decision_tree_rules.png")

        except Exception as e:
            print(f"❌ Lỗi khi chạy mô hình Cây Quyết Định: {e}")
