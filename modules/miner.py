import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import silhouette_score, r2_score, mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import datetime as dt
import os



class DataMinerMixin:
    #=========================================================================
    # THUẬT TOÁN 1: K-MEANS — Gom cụm khách hàng (Cơ sở Toán học)
    # =========================================================================
    def cluster_customers(self, n_clusters=3):
        print("\n" + "="*70)
        print(f"🧠 [K-MEANS] GOM CỤM KHÁCH HÀNG DỰA TRÊN KHOẢNG CÁCH KHÔNG GIAN")
        print("="*70)
        
        try:
            if not hasattr(self, 'rfm_df') or self.rfm_df.empty:
                print("❌ Lỗi: Chưa có dữ liệu RFM. Hãy chạy calculate_rfm() trước.")
                return

            print("--- [BƯỚC 1] CHUẨN HÓA KHÔNG GIAN (Z-SCORE STANDARDIZATION) ---")
            print("  📐 Công thức: Z = (X - μ) / σ")
            print("  💡 Ý nghĩa: Đưa các biến Recency, Frequency, Monetary về cùng thang đo (trung bình = 0, độ lệch chuẩn = 1) để cột tiền (Monetary) không làm lu mờ các cột khác.")
            print("  ⚙️ Thực thi: Đang chạy StandardScaler...")
            
            scaler = StandardScaler()
            rfm_scaled = scaler.fit_transform(self.rfm_df[['Recency', 'Frequency', 'Monetary']])
            print("  ✅ Hoàn tất chuẩn hóa.\n")

            print("--- [BƯỚC 2] ĐO KHOẢNG CÁCH (EUCLIDEAN DISTANCE) ---")
            print("  📐 Công thức: d(p, q) = √[Σ(p_i - q_i)²]")
            print(f"  💡 Ý nghĩa: Đo khoảng cách thực tế giữa từng khách hàng và {n_clusters} tâm cụm (centroids) trong không gian 3 chiều (R, F, M).")
            print(f"  ⚙️ Thực thi: Đang khởi tạo {n_clusters} tâm cụm ngẫu nhiên và gán khách hàng vào tâm cụm gần nhất...\n")

            print("--- [BƯỚC 3] TỐI ƯU TÂM CỤM (CENTROID UPDATE) ---")
            print("  📐 Công thức: μ_k = (1/|C_k|) * Σ(x_i) với x_i ∈ C_k")
            print("  💡 Ý nghĩa: Cập nhật lại vị trí tâm cụm bằng cách lấy trung bình tọa độ của tất cả khách hàng trong cụm đó. Lặp lại đến khi hội tụ (tâm không đổi).")
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)
            
            print("  ⚙️ Thực thi: Thuật toán đã hội tụ thành công!\n")
            
            print("--- [KẾT QUẢ] PHÂN BỐ CỤM ---")
            cluster_counts = self.rfm_df['Cluster'].value_counts().sort_index()
            for cluster, count in cluster_counts.items():
                print(f"  ► Cụm {cluster}: {count} khách hàng")
            print("="*70 + "\n")

        except Exception as e:
            print(f"❌ Lỗi khi chạy K-Means: {e}")

    # =========================================================================
    # THUẬT TOÁN 2: APRIORI — Tìm luật mua chung (Association Rules)
    # Trình bày theo cơ sở Toán học (Support, Confidence, Lift)
    # =========================================================================
    def mine_association_rules(self, min_support=0.5, min_confidence=0.2):
        print("\n" + "="*70)
        print("🛒 [APRIORI] TÌM LUẬT MUA CHUNG DỰA TRÊN CƠ SỞ TOÁN HỌC")
        print("="*70)
        
        try:
            if not hasattr(self, 'basket_df') or self.basket_df.empty:
                print("❌ Lỗi: Chưa có dữ liệu giỏ hàng. Hãy chạy prepare_basket_data() trước.")
                return

            N = self.basket_df.shape[0]
            tong_so_true = self.basket_df.sum().sum()
            print(f"📊 Dữ liệu đầu vào: Ma trận giỏ hàng với N = {N} hóa đơn (tổng cộng {tong_so_true} giao dịch hợp lệ).\n")

            if tong_so_true == 0:
                print("❌ Giỏ hàng trống trơn. Lỗi nằm ở bước tạo giỏ hàng!")
                return

            # ---------------------------------------------------------
            # BƯỚC 1: TÍNH SUPPORT
            # ---------------------------------------------------------
            print("--- [BƯỚC 1] TÍNH SUPPORT (Độ phổ biến) ---")
            print("  📐 Công thức Toán học:")
            print("     Support(A) = Số hóa đơn chứa A / Tổng số hóa đơn (N)")
            print("     Support(A → B) = Số hóa đơn chứa cả A và B / Tổng số hóa đơn (N)")
            print(f"  ⚙️ Thực thi: Đang lọc các mặt hàng/nhóm mặt hàng đạt min_support >= {min_support}...")
            
            frequent_itemsets = apriori(self.basket_df, min_support=min_support, use_colnames=True)
            
            if frequent_itemsets.empty:
                print("  ⚠️ Thất bại: Không có mặt hàng nào đạt chuẩn Support đầu vào.\n")
                self.rules = pd.DataFrame()
                return
            print(f"  ✅ Kết quả: Tìm thấy {len(frequent_itemsets)} nhóm mặt hàng phổ biến.\n")

            # ---------------------------------------------------------
            # BƯỚC 2: TÍNH CONFIDENCE
            # ---------------------------------------------------------
            print("--- [BƯỚC 2] TÍNH CONFIDENCE (Độ tin cậy) ---")
            print("  📐 Công thức Toán học:")
            print("     Confidence(A → B) = Support(A ∪ B) / Support(A)")
            print("  💡 Ý nghĩa: Trong những người mua Sản phẩm A (Antecedents), có bao nhiêu phần trăm mua thêm Sản phẩm B (Consequents).")
            print(f"  ⚙️ Thực thi: Đang sinh các luật kết hợp (Rules) đạt min_confidence >= {min_confidence}...")
            
            self.rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
            
            if self.rules.empty:
                print("  ⚠️ Thất bại: Các mặt hàng không bao giờ được mua chung đủ nhiều (thiếu Confidence).\n")
                return
            print(f"  ✅ Kết quả: Đã sinh ra được {len(self.rules)} luật kết hợp thỏa mãn.\n")

            # ---------------------------------------------------------
            # BƯỚC 3: TÍNH LIFT
            # ---------------------------------------------------------
            print("--- [BƯỚC 3] TÍNH LIFT (Độ nâng) ---")
            print("  📐 Công thức Toán học:")
            print("     Lift(A → B) = Confidence(A → B) / Support(B)")
            print("  💡 Ý nghĩa: ")
            print("     - Lift > 1: Việc mua A kích thích việc mua B (Có sự liên kết dương).")
            print("     - Lift = 1: A và B hoàn toàn độc lập, mua hay không chả liên quan.")
            print("     - Lift < 1: A và B khắc nhau (ví dụ: mua pepsi thì thôi mua coca).")
            print("  ⚙️ Thực thi: Sắp xếp các luật theo chỉ số Lift giảm dần để chọn ra luật mạnh nhất...")
            
            self.rules = self.rules.sort_values('lift', ascending=False)
            
            print(f"  ✅ THÀNH CÔNG: Hoàn tất trích xuất {len(self.rules)} quy luật mua chung tốt nhất!")
            print("="*70 + "\n")

        except Exception as e:
            print(f"❌ Lỗi khi chạy Apriori: {e}")

    # =========================================================================
    # THUẬT TOÁN 3 (V3): LINEAR REGRESSION — DỰ BÁO DOANH THU 
    # Đã vá lỗi: Tự tính TotalAmount và TẠO DỮ LIỆU DỰ BÁO (forecast_df)
    # =========================================================================
    def forecast_sales_v3(self, date_col='InvoiceDate', amount_col='TotalAmount', periods=30):
        print("\n" + "="*70)
        print("📈 [LINEAR REGRESSION] DỰ BÁO DOANH THU (BẢN CHUẨN THỜI GIAN & TƯƠNG LAI)")
        print("="*70)

        try:
            # 0. Tự động tính TotalAmount nếu dữ liệu Siêu thị chưa có
            if amount_col not in self.df.columns:
                if 'UnitPrice' in self.df.columns:
                    qty = self.df['Quantity'] if 'Quantity' in self.df.columns else 1
                    self.df[amount_col] = self.df['UnitPrice'] * qty
                    print("  ✔️ Đã tự động tính cột Doanh thu (TotalAmount) từ UnitPrice.")
                else:
                    print(f"❌ Lỗi: Không tìm thấy cột {amount_col} hay UnitPrice để tính doanh thu.")
                    return

            # --- BƯỚC 1: XỬ LÝ TRỤC THỜI GIAN ---
            print("--- [BƯỚC 1] LẤP ĐẦY NGÀY TRỐNG ---")
            daily_sales = self.df.groupby(pd.to_datetime(self.df[date_col]).dt.date)[amount_col].sum().reset_index()
            daily_sales.columns = ['Date', 'Sales']
            daily_sales['Date'] = pd.to_datetime(daily_sales['Date'])
            daily_sales = daily_sales.set_index('Date')
            
            # Trải phẳng ngày liên tục
            idx = pd.date_range(daily_sales.index.min(), daily_sales.index.max())
            daily_sales = daily_sales.reindex(idx, fill_value=0).reset_index()
            daily_sales.columns = ['Date', 'Sales']
            daily_sales['DayIndex'] = np.arange(len(daily_sales))
            
            X = daily_sales[['DayIndex']]
            y = daily_sales['Sales']

            # --- BƯỚC 2 & 3: CHIA DỮ LIỆU & TRAIN ---
            split_idx = int(len(daily_sales) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Đánh giá lỗi
            mse = mean_squared_error(y_test, y_pred_test)
            rmse = np.sqrt(mse)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            
            print(f"  [Đối chiếu RMSE: Train = {train_rmse:,.0f}  vs  Test = {rmse:,.0f}]")
            if train_rmse < (rmse * 0.5):
                print("  ⚠️ OVERFITTING: Train rất tốt, Test rất tệ.")
            else:
                print("  ✅ MÔ HÌNH ỔN ĐỊNH: Lỗi Train và Test tương đồng.")
                
            # --- [QUAN TRỌNG NHẤT BỊ THIẾU Ở BẢN TRƯỚC]: BƯỚC DỰ BÁO ---
            print(f"--- [BƯỚC 5] DỰ BÁO TƯƠNG LAI CHO {periods} NGÀY TỚI ---")
            model_full = LinearRegression()
            model_full.fit(X, y) # Học trên toàn bộ dữ liệu để bắt xu hướng mới nhất
            
            last_day_index = daily_sales['DayIndex'].max()
            future_indices = np.arange(last_day_index + 1, last_day_index + 1 + periods).reshape(-1, 1)
            future_sales = model_full.predict(pd.DataFrame(future_indices, columns=['DayIndex']))
            
            last_date = daily_sales['Date'].max()
            future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, periods + 1)]
            
            # TẠO BIẾN FORECAST_DF ĐỂ HÀM VẼ CÓ DỮ LIỆU MÀ VẼ
            self.forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted_Sales': future_sales})
            self.daily_sales_history = daily_sales 
            
            print(f"  ✅ Đã lưu dữ liệu dự báo vào self.forecast_df. Sẵn sàng để vẽ đồ thị!")
            print("="*70 + "\n")

        except Exception as e:
            print(f"❌ Lỗi khi dự báo doanh thu: {e}")

    # =========================================================================
    # THUẬT TOÁN 4: DECISION TREE — Trích luật phân loại (Cơ sở Toán học)
    # =========================================================================
    def classify_customers(self):
        print("\n" + "="*70)
        print("🌳 [DECISION TREE] TRÍCH XUẤT LUẬT PHÂN LỚP DỰA TRÊN ĐỘ THUẦN NHẤT")
        print("="*70)

        try:
            if not hasattr(self, 'rfm_df') or 'Cluster' not in self.rfm_df.columns:
                print("❌ Lỗi: Chưa có dữ liệu phân cụm. Vui lòng chạy K-Means trước.")
                return

            X = self.rfm_df[['Recency', 'Frequency', 'Monetary']]
            y = self.rfm_df['Cluster']

            print("--- [BƯỚC 1] TÍNH TOÁN ĐỘ VẨN ĐỤC (GINI IMPURITY) ---")
            print("  📐 Công thức: Gini = 1 - Σ(p_i)²")
            print("     - p_i: Xác suất một khách hàng thuộc về cụm i trong một nút của cây.")
            print("  💡 Ý nghĩa: Thuật toán sẽ tìm cách cắt dữ liệu bằng các điều kiện (ví dụ: Recency < 30) sao cho chỉ số Gini sau khi cắt là thấp nhất (tức là các nhóm được phân ra càng thuần nhất càng tốt).")
            print("  ⚙️ Thực thi: Đang đánh giá mọi ngưỡng chia có thể trên các cột R, F, M...\n")

            print("--- [BƯỚC 2] TỐI ĐA HÓA ĐỘ LỢI THÔNG TIN (INFORMATION GAIN) ---")
            print("  📐 Công thức: Gain = Gini_cha - (Trọng_số_trái * Gini_trái + Trọng_số_phải * Gini_phải)")
            print("  💡 Ý nghĩa: Cây ưu tiên chọn những câu hỏi Yes/No mang lại độ phân tách nhãn (Cluster) rõ rệt nhất ở mỗi nút phân nhánh.")
            
            dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=42)
            dt_classifier.fit(X, y)
            accuracy = dt_classifier.score(X, y)
            
            print("  ⚙️ Thực thi: Đã xây dựng xong cấu trúc cây quyết định (giới hạn max_depth=3 để tránh Overfitting).\n")

            print("--- [BƯỚC 3] ĐÁNH GIÁ & TRÍCH XUẤT LUẬT MỆNH ĐỀ (IF-ELSE) ---")
            print("  💡 Ý nghĩa: Chuyển đổi mô hình toán học dạng cây thành các luật kinh doanh (Business Rules) để Marketing có thể đọc hiểu và sử dụng dễ dàng.")
            print(f"  🎯 Độ chính xác mô phỏng so với K-Means: {accuracy*100:.2f}%")
            
            plt.figure(figsize=(16, 10))
            plot_tree(dt_classifier, 
                      feature_names=['Recency (Ngày)', 'Frequency (Lần)', 'Monetary (Tiền)'],  
                      class_names=[f'Cụm {i}' for i in sorted(y.unique())], 
                      filled=True, rounded=True, fontsize=12)
            plt.title(f"Sơ đồ Cây Quyết Định (Gini Index) - Accuracy: {accuracy*100:.2f}%", fontsize=18, fontweight='bold')
            plt.tight_layout()
            
            os.makedirs('figures', exist_ok=True)
            plt.savefig('figures/decision_tree_rules.png', dpi=300)
            plt.close()
            
            print("  ✅ Kết quả: Đã kết xuất đồ thị toán học của Cây Quyết Định ra file: figures/decision_tree_rules.png")
            print("="*70 + "\n")

        except Exception as e:
            print(f"❌ Lỗi khi chạy mô hình Cây Quyết Định: {e}")

    def evaluate_model_validity(self):
        print("\n" + "="*70)
        print("📊 KIỂM ĐỊNH TÍNH THỰC TẾ CỦA CÁC MÔ HÌNH THUẬT TOÁN")
        print("="*70)
        # Bộ nhớ tạm để in bảng tổng kết cuối
        summary = []
        # ── Bước 1: Kiểm tra điều kiện tiên quyết ──────────────────────────
        print("\n[Bước 1] Kiểm tra điều kiện tiên quyết cho từng mô hình...")
        has_kmeans = hasattr(self, 'rfm_df') and 'Cluster' in self.rfm_df.columns
        has_regression = hasattr(self, 'daily_sales_history') and not self.daily_sales_history.empty
        print(f"   {'✔' if has_kmeans else '✘'} K-Means / Decision Tree : {'rfm_df có cột Cluster — sẵn sàng' if has_kmeans else 'THIẾU — hãy chạy cluster_customers() trước'}")
        print(f"   {'✔' if has_regression else '✘'} Linear Regression      : {'daily_sales_history tồn tại — sẵn sàng' if has_regression else 'THIẾU — hãy chạy forecast_sales() trước'}")
        # ── Bước 2: Kiểm định K-Means (Silhouette Score) ───────────────────
        if has_kmeans:
            X_km = self.rfm_df[['Recency', 'Frequency', 'Monetary']]
            y_km = self.rfm_df['Cluster']

            print("\n" + "-"*70)
            print("2. KIỂM ĐỊNH K-MEANS CLUSTERING")
            print("-"*70)
            print(f"   ĐẦU VÀO : X  = ma trận RFM ({X_km.shape[0]} khách hàng × 3 đặc trưng: Recency, Frequency, Monetary)")
            print(f"   ĐẦU VÀO : y  = nhãn cụm 'Cluster' — {y_km.nunique()} cụm, phân bố: {dict(y_km.value_counts().sort_index())}")
            print(f"   [Bước 2a] Tính Silhouette Score (sample_size=10.000 để tránh tràn RAM)...")

            try:

                sample_size = min(10000, len(X_km))
                sil_score = silhouette_score(X_km, y_km, sample_size=sample_size, random_state=42)

                if sil_score > 0.5:
                    nhan_xet = f"TỐT — Các cụm phân tách rõ ràng. Chiến dịch Marketing cho từng nhóm KH sẽ không bị chồng chéo."
                    ket_luan = "✅ TỐT"
                elif sil_score >= 0.25:
                    nhan_xet = f"TRUNG BÌNH — Ranh giới cụm còn mờ. Cân nhắc thêm đặc trưng hoặc thử K khác."
                    ket_luan = "⚠️ TRUNG BÌNH"
                else:
                    nhan_xet = f"YẾU — Các cụm gần như chồng lên nhau. Nên chạy find_optimal_k() để tìm K phù hợp hơn."
                    ket_luan = "❌ YẾU"

                print(f"   [Bước 2b] Đánh giá ngưỡng Silhouette (> 0.5 tốt / 0.25–0.5 trung bình / < 0.25 kém)...")
                print(f"   ► Metric  : Silhouette Score = {sil_score:.4f}")
                print(f"   ► Kết luận: {nhan_xet}")
                summary.append(("K-Means", f"Silhouette = {sil_score:.4f}", ket_luan))

            except Exception as e:
                print(f"   ❌ Lỗi khi tính Silhouette Score: {e}")

        # ── Bước 3: Kiểm định Decision Tree (Train/Test Accuracy) ──────────
        if has_kmeans:
            print("\n" + "-"*70)
            print("3. KIỂM ĐỊNH DECISION TREE (PHÂN LỚP)")
            print("-"*70)
            print(f"   ĐẦU VÀO : X  = ma trận RFM (giống K-Means, {X_km.shape[0]} dòng × 3 cột)")
            print(f"   ĐẦU VÀO : y  = nhãn 'Cluster' từ K-Means (đóng vai ground-truth)")
            print(f"   ĐẦU VÀO : Chiến lược chia tập: 80% train / 20% test, stratify=y (giữ tỉ lệ cụm)")

            try:

                print(f"   [Bước 3a] Chia train/test (stratified)...")
                X_train, X_test, y_train, y_test = train_test_split(
                    X_km, y_km, test_size=0.2, random_state=42, stratify=y_km
                )
                print(f"   ✔ Tập train: {len(X_train)} mẫu | Tập test ẩn: {len(X_test)} mẫu")

                print(f"   [Bước 3b] Huấn luyện DecisionTreeClassifier(max_depth=3) trên tập train...")
                dt = DecisionTreeClassifier(max_depth=3, random_state=42)
                dt.fit(X_train, y_train)

                print(f"   [Bước 3c] Dự báo nhãn cụm trên tập test ẩn (mô hình chưa từng thấy)...")
                y_pred_dt = dt.predict(X_test)

                print(f"   [Bước 3d] Tính Accuracy = số dự báo đúng / tổng số mẫu test...")
                acc = accuracy_score(y_test, y_pred_dt)

                if acc >= 0.90:
                    nhan_xet = f"XUẤT SẮC — Cây quyết định học được gần hoàn toàn quy tắc từ K-Means. Có thể triển khai thay thế K-Means để phân loại KH mới."
                    ket_luan = "✅ XUẤT SẮC"
                elif acc >= 0.75:
                    nhan_xet = f"TỐT — Cây nắm được phần lớn quy tắc. Một số trường hợp biên giới vẫn còn nhầm lẫn."
                    ket_luan = "✅ TỐT"
                else:
                    nhan_xet = f"CẦN XEM LẠI — Cây chưa học đủ quy tắc từ K-Means. Thử tăng max_depth hoặc kiểm tra lại chất lượng cụm."
                    ket_luan = "⚠️ CẦN XEM LẠI"

                print(f"   [Bước 3e] Đánh giá ngưỡng (≥ 90% xuất sắc / 75–90% tốt / < 75% cần xem lại)...")
                print(f"   ► Metric  : Accuracy trên tập test = {acc*100:.2f}%")
                print(f"   ► Kết luận: {nhan_xet}")
                summary.append(("Decision Tree", f"Accuracy = {acc*100:.2f}%", ket_luan))

            except Exception as e:
                print(f"   ❌ Lỗi khi kiểm định Decision Tree: {e}")

        # ── Bước 4: Kiểm định Linear Regression (R² và RMSE) ───────────────
        if has_regression:

            print("\n" + "-"*70)
            print("4. KIỂM ĐỊNH HỒI QUY TUYẾN TÍNH (LINEAR REGRESSION)")
            print("-"*70)
            print(f"   ĐẦU VÀO : self.daily_sales_history — {len(self.daily_sales_history)} ngày có doanh thu")
            print(f"   ĐẦU VÀO : X  = DayIndex (0 → {len(self.daily_sales_history)-1}) — trục thời gian số hóa")
            print(f"   ĐẦU VÀO : y  = cột 'Sales' — doanh thu thực tế từng ngày")

            try:

                print(f"   [Bước 4a] Xây dựng lại biến X (DayIndex) và y (Sales) từ daily_sales_history...")
                X_lr = np.arange(len(self.daily_sales_history)).reshape(-1, 1)
                y_true_lr = self.daily_sales_history['Sales'].values

                print(f"   [Bước 4b] Huấn luyện lại LinearRegression trên toàn tập lịch sử...")
                lr = LinearRegression()
                lr.fit(X_lr, y_true_lr)
                y_pred_lr = lr.predict(X_lr)
                print(f"   ✔ Hệ số góc = {lr.coef_[0]:,.2f} VNĐ/ngày | Hệ số chặn = {lr.intercept_:,.2f} VNĐ")

                print(f"   [Bước 4c] Tính R² và RMSE...")
                r2   = r2_score(y_true_lr, y_pred_lr)
                rmse = np.sqrt(mean_squared_error(y_true_lr, y_pred_lr))
                mean_sales = np.mean(y_true_lr)
                rmse_pct = (rmse / mean_sales * 100) if mean_sales > 0 else 0

                if r2 > 0.8:
                    nhan_xet_r2 = "TỐT — Xu hướng doanh thu tuyến tính rõ ràng, mô hình đáng tin cậy."
                    ket_luan = "✅ TỐT"
                elif r2 >= 0.5:
                    nhan_xet_r2 = "CHẤP NHẬN — Mô hình nắm được xu hướng chung nhưng dao động thực tế còn lớn."
                    ket_luan = "⚠️ CHẤP NHẬN"
                else:
                    nhan_xet_r2 = "YẾU — Doanh thu biến động phi tuyến, Linear Regression chưa phù hợp. Cân nhắc mô hình phức tạp hơn."
                    ket_luan = "❌ YẾU"

                print(f"   [Bước 4d] Đánh giá ngưỡng (R² > 0.8 tốt / 0.5–0.8 chấp nhận / < 0.5 yếu)...")
                print(f"   ► Metric 1: R-squared (R²)  = {r2:.4f}  → {nhan_xet_r2}")
                print(f"   ► Metric 2: RMSE            = {rmse:,.0f} VNĐ  (~{rmse_pct:.1f}% so với doanh thu trung bình {mean_sales:,.0f} VNĐ/ngày)")
                print(f"   ► Kết luận: Trung bình mỗi ngày, mô hình dự báo lệch khoảng {rmse:,.0f} VNĐ so với thực tế.")
                summary.append(("Linear Regression", f"R² = {r2:.4f} | RMSE = {rmse:,.0f} VNĐ ({rmse_pct:.1f}%)", ket_luan))

            except Exception as e:
                print(f"   ❌ Lỗi khi kiểm định Linear Regression: {e}")

        # ── Bước 5: Bảng tổng kết ──────────────────────────────────────────
        if summary:
            print("\n" + "="*70)
            print("5. BẢNG TỔNG KẾT KIỂM ĐỊNH")
            print("="*70)
            print(f"   {'Mô hình':<22} {'Metric chính':<45} {'Kết luận'}")
            print("   " + "-"*66)
            for ten_mo_hinh, metric, ket_luan in summary:
                print(f"   {ten_mo_hinh:<22} {metric:<45} {ket_luan}")

        print("\n" + "="*70)

    # =========================================================================
    # HÀM TÌM K TỐI ƯU: Elbow Method & Silhouette Score
    # =========================================================================
    def find_optimal_k(self):
        print("\n" + "="*70)
        print("🔍 ĐANG CHẠY THUẬT TOÁN TÌM K TỐI ƯU (ELBOW METHOD & SILHOUETTE)")
        print("="*70)
        print("   [Bước 1] Lấy ma trận đặc trưng RFM từ self.rfm_df...")
        X_rfm = self.rfm_df[['Recency', 'Frequency', 'Monetary']]
        print(f"   ✔ X_rfm shape = {X_rfm.shape} — {X_rfm.shape[0]} khách hàng × 3 đặc trưng")
        wcss = []
        sil_scores = []
        K_range = range(2, 6)
        print("\n   [Bước 2] Tính WCSS và Silhouette Score cho từng K (vui lòng đợi)...\n")
        for k in K_range:

            kmeans = KMeans(n_clusters=k, random_state=42)

            kmeans.fit(X_rfm)

            wcss_value = kmeans.inertia_

            wcss.append(wcss_value)

            sil_value = silhouette_score(X_rfm, kmeans.labels_, sample_size=10000, random_state=42)

            sil_scores.append(sil_value)

            print(f"   ► K = {k} | Độ nhiễu WCSS = {wcss_value:,.0f} | Silhouette Score = {sil_value:.4f}")

        print(f"\n   [Bước 3] Vẽ biểu đồ Elbow Method (WCSS theo K)...")
        plt.figure(figsize=(10, 5))
        plt.plot(K_range, wcss, marker='o', linestyle='--', color='r')
        plt.title('Phương pháp Khuỷu tay (Elbow Method) để tìm K tối ưu', fontsize=14, fontweight='bold')
        plt.xlabel('Số lượng cụm (K)', fontsize=12)
        plt.ylabel('Độ nhiễu WCSS', fontsize=12)
        plt.xticks(K_range)
        plt.grid(True, linestyle=':', alpha=0.7)
        print(f"   [Bước 4] Lưu biểu đồ ra file PNG...")
        os.makedirs('figures', exist_ok=True)
        plt.savefig('figures/elbow_method.png', dpi=300)
        plt.close()
        print("\n✅ Đã xuất biểu đồ Elbow chứng minh K tối ưu tại: figures/elbow_method.png")
        print("="*70 + "\n")