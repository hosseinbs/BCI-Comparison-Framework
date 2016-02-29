from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
import sys

RF_size = 1
slash = '\\'

current_data_dir = sys.argv[1] 
file_name = sys.argv[2]
y_file_name = sys.argv[3]
subject = sys.argv[4]
results_dir_full_path = sys.argv[5]

x_full_path = current_data_dir + slash + file_name
y_full_path = current_data_dir + slash + y_file_name

if __name__ == '__main__':
	
	X = np.loadtxt(x_full_path)
	X = np.asarray(X, dtype=np.float32, order='F')
	Y = np.loadtxt(y_full_path)
	Y = np.asarray(Y, dtype=np.short, order='F')

	clf = RandomForestClassifier(n_estimators=RF_size, oob_score=True, n_jobs= -1, compute_importances = True, min_samples_split=1)
	clf = clf.fit(X.T, Y)

	print x_full_path
	print clf.oob_score_

	fh = open(results_dir_full_path + subject, 'w')
	pickler = pickle.Pickler(fh)
	pickler.dump(clf)
