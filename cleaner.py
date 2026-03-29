import pandas as pd
import numpy as np

class DataCleanerMixin:
     #Xử lý trùng lặp
    def handle_duplicates(self):
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        removed = before - len(self.df)
        self.stats["duplicates_removed"] += removed
        print(f"🧹 [Trùng lặp] Đã xóa {removed} dòng")

    #Xử lý nhiễu
    def handle_noise_and_format(self, exclude_cols=[]):
        cat_cols = self.df.select_dtypes(include=['object']).columns
        noisy_count = 0

        for col in cat_cols:
            if col in exclude_cols:
                continue

            before = self.df[col].copy()
            self.df[col] = self.df[col].str.strip()

            cleaned = self.df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            temp_numeric = pd.to_numeric(cleaned, errors='coerce')

            changed = (~before.fillna("").eq(self.df[col].fillna(""))).sum()
            noisy_count += changed

            if temp_numeric.notna().sum() > 0.8 * len(self.df):
                self.df[col] = temp_numeric

        self.stats["noisy_values_cleaned"] += noisy_count
        print(f"🧹 [Nhiễu] Đã làm sạch ~{noisy_count} giá trị")

    #Xử lý không nhất quán
    def handle_inconsistent_data(self, exclude_cols=[]):
        cat_cols = self.df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if col in exclude_cols: continue
            self.df[col] = self.df[col].str.title()
        print("🧹 [Nhất quán] Đã chuyển hóa Title Case")

    #Xử lý thiếu
    def handle_missing_values(self, exclude_cols=[]):
        before_missing = self.df.isnull().sum().sum()
        self.df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if col in exclude_cols: continue
            self.df[col] = self.df[col].fillna(self.df[col].median())
        cat_cols = self.df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            val = self.df[col].mode()[0] if not self.df[col].mode().empty else "Unknown"
            self.df[col] = self.df[col].fillna(val)

        after_missing = self.df.isnull().sum().sum()
        filled = before_missing - after_missing
        self.stats["missing_filled"] += filled

        print(f"🧹 [Thiếu] Đã điền {filled} giá trị thiếu")

    #Xử lý ngoại lai
    def handle_outliers(self, exclude_cols=[]):
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        outlier_count = 0
        for col in num_cols:
            if col in exclude_cols: continue
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            mask = (self.df[col] < lower) | (self.df[col] > upper)
            outlier_count += mask.sum()
            self.df[col] = np.where(self.df[col] < lower, lower, self.df[col])
            self.df[col] = np.where(self.df[col] > upper, upper, self.df[col])
        self.stats["outliers_capped"]  += outlier_count
        print("🧹 [Ngoại lai] Đã xử lý IQR")

