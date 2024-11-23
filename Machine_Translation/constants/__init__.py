# Step 1- Data ingestion

Data_Path='artifacts/english-to-hindi/data/enTohin.parquet'
engine="pyarrow"
test_size=0.2
random_state=42
val_size=0.1
source="english_sentence"
target="hindi_sentence"

# Step 1- Data preprocessing

special_tokens = ["<sos>", "<eos>"]
max_seq_len=20
filters=''
lower=True
oov_token="<unk>"
start_token="<sos> "
end_token=" <eos>"
post_padding="post"