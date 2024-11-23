# Step 1- Data ingestion

Data_Path='artifacts/english-to-hindi/data/enTohin.parquet'
engine="pyarrow"
test_size=0.2
random_state=42
val_size=0.1
source="english_sentence"
target="hindi_sentence"
