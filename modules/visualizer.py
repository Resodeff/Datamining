import matplotlib.pyplot as plt
import seaborn as sns
import os

class DataVisualizerMixin:
    
    def __init__(self):
        sns.set_theme(style="whitegrid")
    
    #Vẽ biểu đồ phân cụm khách hàng (RFM - KMeans)
    def plot_customer_clusters(self, output_dir="figures"):
        print("📊 Đang vẽ biểu đồ Phân cụm khách hàng (RFM)...")
        if not hasattr(self, 'rfm_df') or 'Cluster' not in self.rfm_df.columns:
            print("⚠️ Không tìm thấy dữ liệu cụm khách hàng để vẽ.")
            return

        os.makedirs(output_dir, exist_ok=True)
        
        # Vẽ biểu đồ 3D cho 3 chiều R, F, M
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(self.rfm_df['Recency'], 
                             self.rfm_df['Frequency'], 
                             self.rfm_df['Monetary'], 
                             c=self.rfm_df['Cluster'], cmap='viridis', s=50, alpha=0.6)
        
        ax.set_title('Phân cụm Khách hàng theo RFM')
        ax.set_xlabel('Recency (Số ngày chưa mua)')
        ax.set_ylabel('Frequency (Số lần mua)')
        ax.set_zlabel('Monetary (Tổng tiền chi)')
        
        # Thêm chú thích
        legend = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend)
        
        plt.savefig(f"{output_dir}/customer_clusters_3d.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Đã lưu biểu đồ phân cụm tại: {output_dir}/customer_clusters_3d.png")

    #Vẽ biểu đồ Quy luật mua hàng (Apriori)
    def plot_association_rules(self, output_dir="figures"):
        print("📊 Đang vẽ biểu đồ Quy luật mua hàng (Support vs Confidence)...")
        if not hasattr(self, 'rules') or self.rules.empty:
            print("⚠️ Không tìm thấy luật kết hợp nào để vẽ.")
            return

        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        
        # Vẽ Scatter plot: Trục X là Support, Trục Y là Confidence, Màu sắc/Kích thước là Lift
        scatter = plt.scatter(self.rules['support'], self.rules['confidence'], 
                              c=self.rules['lift'], s=self.rules['lift']*50, 
                              cmap='Reds', alpha=0.7, edgecolors='k')
        
        plt.colorbar(scatter, label='Lift (Độ nâng)')
        plt.title('Bản đồ Quy luật mua hàng (Association Rules)')
        plt.xlabel('Support (Độ phổ biến)')
        plt.ylabel('Confidence (Độ tin cậy)')
        
        plt.savefig(f"{output_dir}/association_rules_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Đã lưu biểu đồ quy luật tại: {output_dir}/association_rules_scatter.png")

    #Vẽ biểu đồ Dự báo doanh thu (Linear Regression)
    def plot_sales_forecast(self, output_dir="figures"):
        print("📊 Đang vẽ biểu đồ Dự báo doanh thu (Bản chuẩn báo cáo)...")
        if not hasattr(self, 'forecast_df') or not hasattr(self, 'daily_sales_history'):
            print("⚠️ Không tìm thấy dữ liệu dự báo để vẽ.")
            return

        import os
        import matplotlib.pyplot as plt

        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(14, 6))
        
        # --- 🚨 FIX LỖI Ở ĐÂY: Tự động tìm đúng tên cột thời gian ---
        # Kiểm tra xem cột thời gian đang tên là 'InvoiceDate' hay 'Date'
        hist_date_col = 'InvoiceDate' if 'InvoiceDate' in self.daily_sales_history.columns else 'Date'
        fc_date_col = 'InvoiceDate' if 'InvoiceDate' in self.forecast_df.columns else 'Date'
        
        # Kiểm tra cột doanh thu là 'Sales' hay 'Revenue'
        hist_sales_col = 'Sales' if 'Sales' in self.daily_sales_history.columns else self.daily_sales_history.columns[1]
        fc_sales_col = 'Predicted_Sales' if 'Predicted_Sales' in self.forecast_df.columns else self.forecast_df.columns[1]

        # --- BƯỚC 1: CHIA LẠI DỮ LIỆU ĐỂ HIỂN THỊ TRAIN / TEST ---
        split_idx = int(len(self.daily_sales_history) * 0.8)
        train_data = self.daily_sales_history.iloc[:split_idx]
        test_data = self.daily_sales_history.iloc[split_idx:]
        
        # --- BƯỚC 2: VẼ DỮ LIỆU THỰC TẾ ---
        # 1. Tập Train (Màu xanh)
        plt.plot(train_data[hist_date_col], train_data[hist_sales_col], 
                 label='Thực tế (Tập Train - 80%)', color='royalblue', alpha=0.7, linewidth=1.5)
        
        # 2. Tập Test (Màu cam)
        plt.plot(test_data[hist_date_col], test_data[hist_sales_col], 
                 label='Thực tế (Tập Test - 20%)', color='darkorange', alpha=0.9, linewidth=1.5)
        
        # --- BƯỚC 3: VẼ ĐƯỜNG DỰ BÁO ---
        plt.plot(self.forecast_df[fc_date_col], self.forecast_df[fc_sales_col], 
                 label='Đường xu hướng (Dự báo)', color='red', linestyle='--', linewidth=2.5)
        
        # --- BƯỚC 4: TRANG TRÍ BIỂU ĐỒ ---
        # Vẽ ranh giới (Dùng .values[0] để tránh lỗi index)
        split_date = test_data[hist_date_col].values[0]
        plt.axvline(x=split_date, color='grey', linestyle=':', linewidth=2, label='Ranh giới Train / Test')
        
        plt.title('Đánh giá Mô hình Hồi quy và Dự báo Xu hướng Doanh thu', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Thời gian', fontsize=12, fontweight='bold')
        plt.ylabel('Doanh thu (VNĐ)', fontsize=12, fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.5) 
        plt.legend(loc='upper left', fontsize=11, framealpha=0.9)
        plt.tight_layout()
        
        # Xuất file và hiển thị
        plt.savefig(f"{output_dir}/sales_forecast_line.png", dpi=300)
        plt.close()
        print(f"✅ Đã lưu biểu đồ dự báo tại: {output_dir}/sales_forecast_line.png")

    #In báo cáo tổng kết ra
    def print_summary(self):
        print("\n" + "="*70)
        print("🎉 BÁO CÁO TỔNG KẾT KHAI PHÁ DỮ LIỆU SIÊU THỊ 🎉")
        print("="*70)
        
        # --- 1. Thông tin chung ---
        if hasattr(self, 'stats'):
            print(f"🔹 Kích thước dữ liệu sau xử lý: {self.stats.get('rows_after', 'N/A')} hóa đơn, {self.stats.get('cols_after', 'N/A')} trường thông tin.")
        
        print("-" * 70)
        
        # --- 2. Khai phá Nhóm Khách Hàng (K-Means) ---
        if hasattr(self, 'rfm_df') and 'Cluster' in self.rfm_df.columns:
            print("👥 CHÂN DUNG CÁC NHÓM KHÁCH HÀNG (K-MEANS):")
            # Tính trung bình R, F, M cho từng nhóm
            cluster_summary = self.rfm_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(1)
            cluster_counts = self.rfm_df['Cluster'].value_counts()
            
            for cluster_id in cluster_summary.index:
                r = cluster_summary.loc[cluster_id, 'Recency']
                f = cluster_summary.loc[cluster_id, 'Frequency']
                m = cluster_summary.loc[cluster_id, 'Monetary']
                count = cluster_counts[cluster_id]
                print(f"   🔸 Nhóm {cluster_id} ({count} người): Lần cuối mua cách đây {r} ngày | Mua trung bình {f} lần | Đã chi {m:,.0f} VNĐ")
        
        print("-" * 70)
        
        # --- 3. Khai phá Luật Mua Hàng (Apriori) ---
        if hasattr(self, 'rules') and not self.rules.empty:
            print("🛒 TOP 5 QUY LUẬT MUA KÈM NỔI BẬT NHẤT:")
            top_rules = self.rules.head(5) 
            for idx, row in top_rules.iterrows():
                antecedents = ", ".join(list(row['antecedents']))
                consequents = ", ".join(list(row['consequents']))
                conf = row['confidence'] * 100
                lift = row['lift']
                print(f"   🔸 Nếu mua [{antecedents}] ➡️ Khả năng {conf:.0f}% sẽ mua thêm [{consequents}] (Độ mạnh: {lift:.1f}x)")
        
        print("-" * 70)

        # --- 4. Dự báo tương lai (Regression) ---
        if hasattr(self, 'forecast_df') and not self.forecast_df.empty:
            print("📈 DỰ BÁO DOANH THU 3 NGÀY TIẾP THEO:")
            top_forecast = self.forecast_df.head(3)
            for idx, row in top_forecast.iterrows():
                date_str = row['Date'].strftime('%d/%m/%Y')
                sales = row['Predicted_Sales']
                # Xử lý nếu thuật toán dự báo số âm (do dữ liệu ít)
                sales_print = max(0, sales) 
                print(f"   🔸 Ngày {date_str}: Dự kiến thu về {sales_print:,.0f} VNĐ")

        print("-" * 70)
        print("📁 Các biểu đồ phân tích trực quan đã được lưu trong thư mục 'figures/'")
        print("======================================================================" + "\n")

    #Báo cáo tình trạng dữ liệu thiếu
    def report_missing_data(self):
        print("\n" + "-"*50)
        print("🔎 KIỂM TRA TÌNH TRẠNG DỮ LIỆU THIẾU")
        print("-"*50)
        
        if not hasattr(self, 'df') or self.df is None:
            print("⚠️ Chưa có dữ liệu để kiểm tra!")
            return
        
        missing_info = self.df.isnull().sum()
        total_missing = missing_info.sum()
        
        if total_missing == 0:
            print("✅ Tuyệt vời! Dữ liệu hoàn hảo, không bị thiếu ô nào.")
        else:
            print(f"⚠️ Phát hiện tổng cộng {total_missing} ô trống trong dữ liệu:")
            for col, count in missing_info.items():
                if count > 0:
                    percent = (count / len(self.df)) * 100
                    print(f"   - Cột '{col}': thiếu {count} giá trị ({percent:.1f}%)")
        print("-"*50 + "\n")