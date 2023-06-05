from zeugma.embeddings import EmbeddingTransformer
import gensim
from numpy import savez_compressed

# GloVe
Glove = EmbeddingTransformer('glove')
############df1
Glove_df1_train = Glove.fit_transform(df1_train)
Glove_df1_test = Glove.transform(df1_test)
############df2
Glove_df2_train = Glove.fit_transform(df2_train)
Glove_df2_test = Glove.transform(df2_test)
############df3
Glove_df3_train = Glove.fit_transform(df3_train)
Glove_df3_test = Glove.transform(df3_test)
############df4
Glove_df4_train = Glove.fit_transform(df4_train)
Glove_df4_test = Glove.transform(df4_test)

# Save matrix Glove
savez_compressed('Glove_df1_train.npz', Glove_df1_train)
savez_compressed('Glove_df2_train.npz', Glove_df2_train)
savez_compressed('Glove_df3_train.npz', Glove_df3_train)
savez_compressed('Glove_df4_train.npz', Glove_df4_train)

savez_compressed('Glove_df1_test.npz', Glove_df1_test)
savez_compressed('Glove_df2_test.npz', Glove_df2_test)
savez_compressed('Glove_df3_test.npz', Glove_df3_test)
savez_compressed('Glove_df4_test.npz', Glove_df4_test)
