from zeugma.embeddings import EmbeddingTransformer
import gensim
from numpy import savez_compressed

# Fasttext
Fasttext = EmbeddingTransformer('fasttext')
############df1
Fasttext_df1_train = Fasttext.fit_transform(df1_train)
Fasttext_df1_test = Fasttext.transform(df1_test)
############df2
Fasttext_df2_train = Fasttext.fit_transform(df2_train)
Fasttext_df2_test = Fasttext.transform(df2_test)
############df3
Fasttext_df3_train = Fasttext.fit_transform(df3_train)
Fasttext_df3_test = Fasttext.transform(df3_test)
############df4
Fasttext_df4_train = Fasttext.fit_transform(df4_train)
Fasttext_df4_test = Fasttext.transform(df4_test)

# Save matrix Fasttext
savez_compressed('Fasttext_df1_train.npz', Fasttext_df1_train)
savez_compressed('Fasttext_df2_train.npz', Fasttext_df2_train)
savez_compressed('Fasttext_df3_train.npz', Fasttext_df3_train)
savez_compressed('Fasttext_df4_train.npz', Fasttext_df4_train)

savez_compressed('Fasttext_df1_test.npz', Fasttext_df1_test)
savez_compressed('Fasttext_df2_test.npz', Fasttext_df2_test)
savez_compressed('Fasttext_df3_test.npz', Fasttext_df3_test)
savez_compressed('Fasttext_df4_test.npz', Fasttext_df4_test)
