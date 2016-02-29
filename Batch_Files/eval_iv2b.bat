echo 'evaluating BCIC iv2b: '

python ../bci_framework/run_bcic.py BCICIV2b LDA BP -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2b LDA wackerman -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2b LDA logbp -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2b LDA morlet -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2b LDA AR -1 ALL evaluate

python ../bci_framework/run_bcic.py BCICIV2b QDA BP -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2b QDA wackerman -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2b QDA logbp -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2b QDA morlet -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2b QDA AR -1 ALL evaluate

python ../bci_framework/run_bcic.py BCICIV2b RANDOM_FOREST BP -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2b RANDOM_FOREST wackerman -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2b RANDOM_FOREST logbp -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2b RANDOM_FOREST morlet -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2b RANDOM_FOREST AR -1 ALL evaluate

python ../bci_framework/run_bcic.py BCICIV2b SVM BP -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2b SVM wackerman -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2b SVM logbp -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2b SVM morlet -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2b SVM AR -1 ALL evaluate

python ../bci_framework/run_bcic.py BCICIV2b LogisticRegression BP -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2b LogisticRegression wackerman -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2b LogisticRegression logbp -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2b LogisticRegression morlet -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2b LogisticRegression AR -1 ALL evaluate

python ../bci_framework/run_bcic.py BCICIV2b Boosting BP -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2b Boosting wackerman -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2b Boosting logbp -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2b Boosting morlet -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2b Boosting AR -1 ALL evaluate

python ../bci_framework/run_bcic.py BCICIV2b MLP BP -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2b MLP wackerman -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2b MLP logbp -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2b MLP morlet -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2b MLP AR -1 ALL evaluate