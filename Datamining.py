import pandas as pd
import os

from modules.cleaner import DataCleanerMixin
from modules.transformer import DataTransformerMixin
from modules.miner import DataMinerMixin
from modules.visualizer import DataVisualizerMixin

class Datamining(DataCleanerMixin, DataTransformerMixin, DataMinerMixin, DataVisualizerMixin):
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.stats = {
        "rows_before": 0,
        "rows_after": 0,
        "cols_before": 0,
        "cols_after": 0,
        "missing_filled": 0,
        "duplicates_removed": 0,
        "noisy_values_cleaned": 0,
        "outliers_capped": 0
        }
        self.income_before = None
        self.income_after = None
        self.scores_before = None
        self.scores_after = None
    
    #load file
    def load_data(self):
        if not os.path.exists(self.file_path):
            print(f"❌ Lỗi: Không tìm thấy file '{self.file_path}'")
            return False
        try:
            if self.file_path.endswith('.xlsx'):
                self.df = pd.read_excel(self.file_path, engine='openpyxl')
            else:
                self.df = pd.read_csv(self.file_path, sep=None, engine='python', encoding='utf-8', on_bad_lines='skip')
            
            self.df.columns = self.df.columns.str.strip().str.lower()
            
            mapping = {
                'member_number': 'InvoiceNo',
                'date': 'InvoiceDate',
                'itemdescription': 'ProductName',
                'invoiceno': 'InvoiceNo',
                'customerid': 'CustomerID',
                'productname': 'ProductName',
                'quantity': 'Quantity',
                'unitprice': 'UnitPrice',
                'invoicedate': 'InvoiceDate'
            }
            self.df = self.df.rename(columns=mapping)
            
            if 'Quantity' in self.df.columns and 'UnitPrice' in self.df.columns:
                self.df['TotalAmount'] = self.df['Quantity'] * self.df['UnitPrice']


            print(f"✅ Đã tải và chuẩn hóa: {self.df.shape[0]} dòng")
            print(f"🔍 Cột hiện tại: {list(self.df.columns)}")
            return True
        except Exception as e:
            print(f"❌ Lỗi đọc file: {e}")
            return False
    
    #Lưu file
    def save_data(self, output_path):
        self.df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"💾 Đã lưu file: {output_path}")

    #hàm chạy dữ liệu
    def run_pipeline(self, output_path, ignore_cols=[], date_cols=[], drop_cols_final=[], min_support=0.05, min_confidence=0.2):
        #Load dữ liệu
        if self.load_data():
            print("🛒 Đang bắt đầu Pipeline phân tích dữ liệu Siêu thị...")
            
            #Báo cáo dữ liệu ban đầu
            self.report_missing_data()

            #Làm sạch dữ liệu
            self.handle_duplicates()
            self.handle_noise_and_format(exclude_cols=ignore_cols + date_cols)
            self.handle_inconsistent_data(exclude_cols=ignore_cols)
            self.handle_missing_values(exclude_cols=ignore_cols + date_cols)
            self.handle_outliers(exclude_cols=ignore_cols)

            print("⚙️ Đang tính toán các chỉ số kinh doanh...")
            #Tạo bộ chỉ số RFM (Recency, Frequency, Monetary)
            self.calculate_rfm() 

            #Chuyển đổi dữ liệu sang dạng giỏ hàng để tìm luật mua chung
            self.prepare_basket_data() 

            print("\n🔍 Đang chạy thực nghiệm để tìm số cụm (K) tối ưu...")
            self.find_optimal_k()

            print("🚀 Đang chạy các thuật toán Khai phá dữ liệu...")
            #Gom cụm khách hàng (Dùng K-Means trên tập RFM)
            self.cluster_customers(n_clusters=3) 

            #Phân lớp theo K-Means
            self.classify_customers()
            
            #Tìm sản phẩm mua cùng nhau (Dùng Apriori/FP-Growth)
            self.mine_association_rules(min_support=min_support, min_confidence=min_confidence) 
            
            #Dự báo doanh thu (Dùng Time Series ARIMA hoặc Regression)
            self.forecast_sales_v3(periods=30) # Dự báo cho 30 ngày (1 tháng) tới

            print("\n📊 Đang đánh giá độ tin cậy của các mô hình...")
            self.evaluate_model_validity()

            # Định dạng lại ngày tháng và xóa cột thừa
            self.finalize_date_format(cols=date_cols)
            self.drop_columns(drop_cols_final)

            # Cập nhật thống kê số dòng/cột sau xử lý
            self.stats["rows_after"] = self.df.shape[0]
            self.stats["cols_after"] = self.df.shape[1]

            # Vẽ biểu đồ đặc thù cho Siêu thị
            print("📊 Đang xuất biểu đồ báo cáo...")
            self.plot_customer_clusters()   # Vẽ cụm khách hàng VIP/Thường
            self.plot_sales_forecast()      # Vẽ đường xu hướng doanh thu
            self.plot_association_rules()   # Vẽ mạng lưới các sản phẩm mua cùng nhau (nếu có)
            
            self.print_summary()
            
            # Lưu file CSV kết quả cuối cùng
            self.save_data(output_path)
            print("✅ Đã hoàn thành toàn bộ tiến trình!")
