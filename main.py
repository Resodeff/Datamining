from Datamining import Datamining

if __name__ == "__main__":
    input_data = "Instacart_Category_Data_2Years_Fixed.csv" 
    cleaner = Datamining(input_data)
    
    datetime_cols = ['InvoiceDate'] 
    drop_cols = [] 

    cleaner.run_pipeline(
        output_path="ket_qua_final.csv",
        ignore_cols=[], 
        date_cols=datetime_cols,
        drop_cols_final=drop_cols,
        min_support=0.5,
        min_confidence=0.5
    )