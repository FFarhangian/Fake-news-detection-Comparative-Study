from zeugma.embeddings import EmbeddingTransformer
import gensim
from numpy import savez_compressed

# Word2Vec
W2V = EmbeddingTransformer('word2vec')
############df1
W2V_df1_train = W2V.fit_transform(df1_train)
W2V_df1_test = W2V.transform(df1_test)
############df2
W2V_df2_train = W2V.fit_transform(df2_train)
W2V_df2_test = W2V.transform(df2_test)
############df3
W2V_df3_train = W2V.fit_transform(df3_train)
W2V_df3_test = W2V.transform(df3_test)
############df4
W2V_df4_train = W2V.fit_transform(df4_train)
W2V_df4_test = W2V.transform(df4_test)

# Save matrix W2V

savez_compressed('W2V_df1_train.npz', W2V_df1_train)
savez_compressed('W2V_df2_train.npz', W2V_df2_train)
savez_compressed('W2V_df3_train.npz', W2V_df3_train)
savez_compressed('W2V_df4_train.npz', W2V_df4_train)

savez_compressed('W2V_df1_test.npz', W2V_df1_test)
savez_compressed('W2V_df2_test.npz', W2V_df2_test)
savez_compressed('W2V_df3_test.npz', W2V_df3_test)
savez_compressed('W2V_df4_test.npz', W2V_df4_test)
