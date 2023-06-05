from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import savez_compressed

# TF-IDF
TFIDF = TfidfVectorizer(analyzer='word', lowercase=True, use_idf=True, stop_words='english')
############df1
TFIDF_df1_train = TFIDF.fit_transform(df1_train)
TFIDF_df1_test = TFIDF.transform(df1_test)
############df2
TFIDF_df2_train = TFIDF.fit_transform(df2_train)
TFIDF_df2_test = TFIDF.transform(df2_test)
############df3
TFIDF_df3_train = TFIDF.fit_transform(df3_train)
TFIDF_df3_test = TFIDF.transform(df3_test)
############df4
TFIDF_df4_train = TFIDF.fit_transform(df4_train)
TFIDF_df4_test = TFIDF.transform(df4_test)

# Save matrix TFIDF
sparse.save_npz("TFIDF_df1_train.npz", TFIDF_df1_train)
sparse.save_npz("TFIDF_df2_train.npz", TFIDF_df2_train)
sparse.save_npz("TFIDF_df3_train.npz", TFIDF_df3_train)
sparse.save_npz("TFIDF_df4_train.npz", TFIDF_df4_train)

sparse.save_npz("TFIDF_df1_test.npz", TFIDF_df1_test)
sparse.save_npz("TFIDF_df2_test.npz", TFIDF_df2_test)
sparse.save_npz("TFIDF_df3_test.npz", TFIDF_df3_test)
sparse.save_npz("TFIDF_df4_test.npz", TFIDF_df4_test)

