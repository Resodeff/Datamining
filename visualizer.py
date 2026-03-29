import pandas as pd
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
        print("📊 Đang vẽ biểu đồ Dự báo doanh thu...")
        if not hasattr(self, 'forecast_df') or not hasattr(self, 'daily_sales_history'):
            print("⚠️ Không tìm thấy dữ liệu dự báo để vẽ.")
            return

        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(14, 6))
        
        # Vẽ đường doanh thu trong quá khứ (Màu xanh, nét liền)
        plt.plot(self.daily_sales_history['Date'], self.daily_sales_history['Sales'], 
                 label='Doanh thu Thực tế', color='blue', marker='o', markersize=4)
        
        # Vẽ đường dự báo trong tương lai (Màu cam, nét đứt)
        plt.plot(self.forecast_df['Date'], self.forecast_df['Predicted_Sales'], 
                 label='Doanh thu Dự báo', color='orange', linestyle='--', marker='x', markersize=4)
        
        plt.title('Dự báo Xu hướng Doanh thu Siêu thị')
        plt.xlabel('Ngày')
        plt.ylabel('Doanh thu')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        
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