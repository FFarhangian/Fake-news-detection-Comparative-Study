from sklearn.feature_extraction.text import CountVectorizer
from numpy import savez_compressed

# TF
TF = CountVectorizer(analyzer='word', lowercase=True, stop_words='english')
############df1
TF_df1_train = TF.fit_transform(df1_train)
TF_df1_test = TF.transform(df1_test)
############df2
TF_df2_train = TF.fit_transform(df2_train)
TF_df2_test = TF.transform(df2_test)
############df3
TF_df3_train = TF.fit_transform(df3_train)
TF_df3_test = TF.transform(df3_test)
############df4
TF_df4_train = TF.fit_transform(df4_train)
TF_df4_test = TF.transform(df4_test)

# Save matrix TF
sparse.save_npz("TF_df1_train.npz", TF_df1_train)
sparse.save_npz("TF_df2_train.npz", TF_df2_train)
sparse.save_npz("TF_df3_train.npz", TF_df3_train)
sparse.save_npz("TF_df4_train.npz", TF_df4_train)

sparse.save_npz("TF_df1_test.npz", TF_df1_test)
sparse.save_npz("TF_df2_test.npz", TF_df2_test)
sparse.save_npz("TF_df3_test.npz", TF_df3_test)
sparse.save_npz("TF_df4_test.npz", TF_df4_test)
