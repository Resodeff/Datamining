import pandas as pd
import numpy as np
import json
import os
import datetime as dt

class DataTransformerMixin:
    #Áp dụng luật
    def apply_rag_law(self, law_path):
        if not os.path.exists(law_path):
            print(f"⚠️ Không tìm thấy file luật '{law_path}'")
            return
        print(f"\n--- Áp dụng luật (RAG) ---")
        with open(law_path, 'r', encoding='utf-8') as f:
            laws = json.load(f)

        for col, law in laws.items():
            if col not in self.df.columns: continue
            
            if "mapping" in law:
                self.df[col] = self.df[col].replace(law['mapping'])
            
            if 'min' in law or 'max' in law:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

            if "min" in law:
                self.df = self.df[self.df[col] >= law['min']]
            if 'max' in law:
                self.df = self.df[self.df[col] <= law['max']]
            if "allowed_values" in law:
                valid = [x.lower() for x in law['allowed_values']]
                mask = self.df[col].fillna("").astype(str).str.lower().isin(valid)
                self.df = self.df[mask]
        print(f"✅ Kết thúc RAG. Dữ liệu còn: {len(self.df)} dòng")

    #Chia nhóm
    def feature_engineering(self):
        inc_col = next((c for c in self.df.columns if 'income' in c.lower()), None)
        if inc_col:
            print(f"Đang xử lý thu nhập trên cột: '{inc_col}'")
            
            self.df[inc_col] = self.df[inc_col].astype(str).str.replace(r'[^\d]', '', regex=True)
            
            self.df[inc_col] = pd.to_numeric(self.df[inc_col], errors='coerce').fillna(0)
                  
            bins = [-1, 4999999, 10000000, float('inf')]
            labels = ['Low', 'Medium', 'High']
            
            self.df['Income_Level'] = pd.cut(self.df[inc_col], bins=bins, labels=labels)
            
            print(f"Đã tạo cột 'Income_Level'.")
            print(self.df[[inc_col, 'Income_Level']].head())
        else:
            print("Không tìm thấy cột Income.")

        age_col = next((c for c in self.df.columns if c.strip().lower() == 'age'), None)
        if age_col:
            print(f"[DEBUG] Đã tìm thấy cột gốc: '{age_col}'")
            
            self.df[age_col] = pd.to_numeric(self.df[age_col], errors='coerce').fillna(0)
            self.df[age_col] = self.df[age_col].abs()
            
            print(f"- Min tuổi: {self.df[age_col].min()}")
            print(f"- Max tuổi: {self.df[age_col].max()}")

            bins = [-1, 19, 22, 25, 100]
            labels = ['Teen', 'Young Adult', 'Adult', 'Other']
            
            try:
                self.df['age_group'] = pd.cut(self.df[age_col], bins=bins, labels=labels)
                print("🎉 [THÀNH CÔNG] Đã tạo cột 'age_group'")
                print(self.df['age_group'].value_counts())
            except Exception as e:
                print(f"[LỖI CHIA NHÓM] Chi tiết lỗi: {e}")   
        else:
            print(f"[LỖI TÊN CỘT] Không tìm thấy cột 'age' trong danh sách: {list(self.df.columns)}")

        target_subjects = ['score_math', 'score_english', 'score_physics']
        found_cols = []
        
        for subject in target_subjects:
            col = next((c for c in self.df.columns if subject in c.lower()), None)
            if col:
                found_cols.append(col)
        
        if len(found_cols) == 3:
            print(f"Đã tìm thấy 3 môn: {found_cols}. Đang tính điểm trung bình...")
            for c in found_cols:
                self.df[c] = pd.to_numeric(self.df[c], errors='coerce').fillna(0)
            
            # Tính trung bình theo hàng (axis=1) và làm tròn 2 số
            self.df['average_score'] = self.df[found_cols].mean(axis=1).round(2)
            
            print(f"Đã tạo cột 'average_score'. Mẫu: {self.df['average_score'].head().tolist()}")
        else:
            print(f"Không tìm đủ 3 môn Math, English, Physics. Tìm được: {found_cols}")

    #Chuẩn hóa Max-Min
    def scale_specific_cols(self, cols_to_scale):
        for col in cols_to_scale:
            if col in self.df.columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val != min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
                    self.df[col] = self.df[col].round(4)
                    print(f"📏 [Scale] Đã chuẩn hóa Min-Max: {col}")

    #chuẩn hóa điểm về [0-1]
    def normalize_subjects(self):
        print("\n" + "="*40)
        print("⚖️ ĐANG CHUẨN HÓA ĐIỂM SỐ VỀ [0-1]")
        print("="*40)

        subjects_keywords = ['score_math', 'score_english', 'score_physics']
        
        target_cols = []
        for kw in subjects_keywords:
            found = next((c for c in self.df.columns if kw in c.lower()), None)
            if found:
                target_cols.append(found)

        if not target_cols:
            print("⚠️ Không tìm thấy cột môn học nào để chuẩn hóa.")
            return

        print(f"👉 Các cột sẽ chuẩn hóa: {target_cols}")

        for col in target_cols:
            min_val = self.df[col].min()
            max_val = self.df[col].max()

            # Đề phòng trường hợp min = max (chia cho 0 sẽ lỗi)
            if max_val - min_val == 0:
                print(f"   ⚠️ Cột '{col}' có giá trị không đổi ({min_val}). Gán về 0.")
                self.df[col] = 0.0
            else:
                # Áp dụng công thức Min-Max Scaling
                self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
                
                self.df[col] = self.df[col].round(2) 
                print(f"   ✅ Đã chuẩn hóa '{col}'. Min cũ: {min_val}, Max cũ: {max_val} -> [0-1]")

        check_cols = target_cols + ['average_score'] if 'average_score' in self.df.columns else target_cols
        print("\n📊 Kết quả sau khi chuẩn hóa (Average vẫn giữ nguyên):")
        print(self.df[check_cols].head())
        print("="*40 + "\n")

    #Chuẩn hóa income = Z-score
    def standardize_income(self):
        print("\n" + "="*40)
        print("📉 CHUẨN HÓA INCOME THEO Z-SCORE")
        print("="*40)

        # 1. Tìm cột Income
        inc_col = next((c for c in self.df.columns if 'income' in c.lower()), None)

        if not inc_col:
            print("⚠️ Không tìm thấy cột Income để chuẩn hóa.")
            return

        self.df[inc_col] = self.df[inc_col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
        self.df[inc_col] = pd.to_numeric(self.df[inc_col], errors='coerce')

        # 3. Tính Mean (μ) và Std (σ)
        mu = self.df[inc_col].mean()
        sigma = self.df[inc_col].std()

        print(f"   📊 Thống kê trước khi chuẩn hóa:")
        print(f"      - Trung bình (Mean): {mu:,.2f}")
        print(f"      - Độ lệch chuẩn (Std): {sigma:,.2f}")

        # 4. Áp dụng công thức Z-score
        if sigma == 0:
            print("   ⚠️ Độ lệch chuẩn = 0 (tất cả thu nhập giống hệt nhau). Gán Z-score = 0.")
            self.df[inc_col] = 0.0
        else:
            self.df[inc_col] = (self.df[inc_col] - mu) / sigma
            
            print(f"   ✅ Đã chuẩn hóa Z-score cho '{inc_col}'.")
            print(f"      - Giá trị mới (Mean ≈ 0, Std ≈ 1):")
            print(self.df[inc_col].head())

        print("="*40 + "\n")

    #Tạo thuộc tính
    def create_advanced_features(self):
        print("\n" + "="*40)
        print("🚀 TẠO CÁC THUỘC TÍNH MỚI (ADVANCED FEATURES)")
        print("="*40)

        math_col = next((c for c in self.df.columns if 'math' in c.lower()), None)
        eng_col = next((c for c in self.df.columns if 'english' in c.lower()), None)
        phy_col = next((c for c in self.df.columns if 'physics' in c.lower()), None)

        if math_col and eng_col and phy_col:
            for col in [math_col, eng_col, phy_col]:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
            
            self.df['Performance_Index'] = (
                0.4 * self.df[math_col] + 
                0.3 * self.df[eng_col] + 
                0.3 * self.df[phy_col]
            ).round(2)
            print(f"   ✅ 1. Đã tạo 'Performance_Index' từ {math_col}, {eng_col}, {phy_col}.")
        else:
            print("   ⚠️ Bỏ qua Performance_Index (thiếu môn học).")

        avg_col = next((c for c in self.df.columns if 'average' in c.lower() and 'score' in c.lower()), None)
        
        if avg_col:
            self.df['Is_Excellent'] = (self.df[avg_col] >= 8.5).astype(int)
            
            count_exc = self.df['Is_Excellent'].sum()
            print(f"   ✅ 2. Đã tạo 'Is_Excellent' dựa trên '{avg_col}'.")
            print(f"      -> Số sinh viên xuất sắc: {count_exc}")
        else:
            print("   ⚠️ Bỏ qua Is_Excellent (chưa có cột average_score).")

        att_col = next((c for c in self.df.columns if 'attendance' in c.lower()), None)  
        if att_col:
            self.df[att_col] = self.df[att_col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            self.df[att_col] = pd.to_numeric(self.df[att_col], errors='coerce').fillna(0)
            
            bins = [-1, 69.99, 84.99, 100]
            labels = ['Low', 'Medium', 'High']
            
            self.df['Attendance_Category'] = pd.cut(self.df[att_col], bins=bins, labels=labels)
            
            print(f"   ✅ 3. Đã tạo 'Attendance_Category' từ '{att_col}'.")
            print(f"      -> Phân bố: {self.df['Attendance_Category'].value_counts().to_dict()}")
        else:
            print("   ⚠️ Bỏ qua Attendance_Category (không tìm thấy cột Attendance).")

        print("="*40 + "\n")

    #Xóa cột
    def drop_columns(self, cols_to_drop=[]):
        existing_cols = [c for c in cols_to_drop if c in self.df.columns]
        if existing_cols:
            self.df.drop(columns=existing_cols, inplace=True)
            print(f"🗑 [Xóa cột] Đã xóa: {existing_cols}") 
        else:
            print("⚠️ [Xóa cột] Không có cột nào cần xóa hoặc cột không tồn tại.")

    #Định dạng format
    def finalize_date_format(self, cols=[]):
        for col in cols:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce', dayfirst=True)
                self.df[col] = self.df[col].dt.strftime('%Y/%m/%d')
                print(f"🗓 [Format] Đã chuyển '{col}' sang YYYY/MM/DD")

    #Hàm tính toán RFM (Phục vụ gom cụm khách hàng)
    def calculate_rfm(self, customer_col='CustomerID', date_col='InvoiceDate', amount_col='TotalAmount'):
        """
        Tính toán 3 chỉ số RFM (Recency, Frequency, Monetary) cho từng khách hàng.
        """
        print("⚙️ Đang tính toán bộ chỉ số RFM...")
        try:
            self.df[date_col] = pd.to_datetime(self.df[date_col])
            
            # Lấy mốc thời gian là ngày mới nhất trong dữ liệu + 1 ngày để tính Recency
            snapshot_date = self.df[date_col].max() + dt.timedelta(days=1)
            
            # Gom nhóm theo Khách hàng và tính RFM
            self.rfm_df = self.df.groupby(customer_col).agg({
                date_col: lambda x: (snapshot_date - x.max()).days, # Recency: Số ngày từ lần mua cuối
                customer_col: 'count',                              # Frequency: Tổng số lượng đơn / sản phẩm
                amount_col: 'sum'                                   # Monetary: Tổng tiền đã chi
            }).rename(columns={
                date_col: 'Recency',
                customer_col: 'Frequency',
                amount_col: 'Monetary'
            })
            
            print(f"✅ Đã tạo thành công tập dữ liệu RFM cho {len(self.rfm_df)} khách hàng.")
        except Exception as e:
            print(f"❌ Lỗi khi tính RFM: {e}")

    # 2. Hàm tạo Giỏ hàng (Phục vụ tìm luật mua chung)
    def prepare_basket_data(self, invoice_col='InvoiceNo', product_col='ProductName', qty_col='Quantity'):
        print("⚙️ Đang đóng gói dữ liệu thành các Giỏ hàng...")
        try:
            if invoice_col not in self.df.columns:
                print(f"❌ Lỗi: Không tìm thấy cột '{invoice_col}'. Các cột hiện có: {list(self.df.columns)}")
                return

            df_filtered = self.df
            basket = (df_filtered.groupby([invoice_col, product_col])[qty_col]
                      .sum().unstack().reset_index().fillna(0)
                      .set_index(invoice_col))
            
            def encode_units(x):
                return 1 if x >= 1 else 0
                
            self.basket_df = basket.map(encode_units).astype(bool)
                     
            print(f"🧹 Sau khi lọc, còn lại {self.basket_df.shape[0]} hóa đơn hợp lệ.")
            print(f"✅ Đã tạo ma trận giỏ hàng xong!")
            print(f"📊 Kiểm tra ma trận: {self.basket_df.sum().sum()} ô có giá trị True")
            
        except Exception as e:
            print(f"❌ Lỗi khi tạo giỏ hàng: {e}")