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

# step 3-get model

embedding_dim = 128
hidden_units = 256

# step 4-model trainer
batch_size=64
epochs=1

# step 4-model prediction
text="how are you"