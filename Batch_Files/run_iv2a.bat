echo 'running BCIC iv2a: '

python ../bci_framework/run_bcic.py BCICIV2a LDA BP -1 ALL run
python ../bci_framework/run_bcic.py BCICIV2a LDA wackerman -1 ALL run
python ../bci_framework/run_bcic.py BCICIV2a LDA logbp -1 ALL run
python ../bci_framework/run_bcic.py BCICIV2a LDA morlet -1 ALL run
python ../bci_framework/run_bcic.py BCICIV2a LDA AR -1 ALL run

python ../bci_framework/run_bcic.py BCICIV2a QDA BP -1 ALL run
python ../bci_framework/run_bcic.py BCICIV2a QDA wackerman -1 ALL run
python ../bci_framework/run_bcic.py BCICIV2a QDA logbp -1 ALL run
python ../bci_framework/run_bcic.py BCICIV2a QDA morlet -1 ALL run
python ../bci_framework/run_bcic.py BCICIV2a QDA AR -1 ALL run

python ../bci_framework/run_bcic.py BCICIV2a RANDOM_FOREST BP -1 ALL run
python ../bci_framework/run_bcic.py BCICIV2a RANDOM_FOREST wackerman -1 ALL run
python ../bci_framework/run_bcic.py BCICIV2a RANDOM_FOREST logbp -1 ALL run
python ../bci_framework/run_bcic.py BCICIV2a RANDOM_FOREST morlet -1 ALL run
python ../bci_framework/run_bcic.py BCICIV2a RANDOM_FOREST AR -1 ALL run

python ../bci_framework/run_bcic.py BCICIV2a SVM BP -1 ALL run
python ../bci_framework/run_bcic.py BCICIV2a SVM wackerman -1 ALL run
python ../bci_framework/run_bcic.py BCICIV2a SVM logbp -1 ALL run
python ../bci_framework/run_bcic.py BCICIV2a SVM morlet -1 ALL run
python ../bci_framework/run_bcic.py BCICIV2a SVM AR -1 ALL run

python ../bci_framework/run_bcic.py BCICIV2a LogisticRegression BP -1 ALL run
python ../bci_framework/run_bcic.py BCICIV2a LogisticRegression wackerman -1 ALL run
python ../bci_framework/run_bcic.py BCICIV2a LogisticRegression logbp -1 ALL run
python ../bci_framework/run_bcic.py BCICIV2a LogisticRegression morlet -1 ALL run
python ../bci_framework/run_bcic.py BCICIV2a LogisticRegression AR -1 ALL run

python ../bci_framework/run_bcic.py BCICIV2a Boosting BP -1 ALL run
python ../bci_framework/run_bcic.py BCICIV2a Boosting wackerman -1 ALL run
python ../bci_framework/run_bcic.py BCICIV2a Boosting logbp -1 ALL run
python ../bci_framework/run_bcic.py BCICIV2a Boosting morlet -1 ALL run
python ../bci_framework/run_bcic.py BCICIV2a Boosting AR -1 ALL run

python ../bci_framework/run_bcic.py BCICIV2a MLP BP -1 ALL run
python ../bci_framework/run_bcic.py BCICIV2a MLP wackerman -1 ALL run
python ../bci_framework/run_bcic.py BCICIV2a MLP logbp -1 ALL run
python ../bci_framework/run_bcic.py BCICIV2a MLP morlet -1 ALL run
python ../bci_framework/run_bcic.py BCICIV2a MLP AR -1 ALL run

echo 'running BCIC iv2a with CSP: '

python ../bci_framework/run_bcic.py BCICIV2a LDA BP 2 CSP run
python ../bci_framework/run_bcic.py BCICIV2a LDA wackerman 2 CSP run
python ../bci_framework/run_bcic.py BCICIV2a LDA logbp 2 CSP run
python ../bci_framework/run_bcic.py BCICIV2a LDA morlet 2 CSP run
python ../bci_framework/run_bcic.py BCICIV2a LDA AR 2 CSP run

python ../bci_framework/run_bcic.py BCICIV2a QDA BP 2 CSP run
python ../bci_framework/run_bcic.py BCICIV2a QDA wackerman 2 CSP run
python ../bci_framework/run_bcic.py BCICIV2a QDA logbp 2 CSP run
python ../bci_framework/run_bcic.py BCICIV2a QDA morlet 2 CSP run
python ../bci_framework/run_bcic.py BCICIV2a QDA AR 2 CSP run

python ../bci_framework/run_bcic.py BCICIV2a RANDOM_FOREST BP 2 CSP run
python ../bci_framework/run_bcic.py BCICIV2a RANDOM_FOREST wackerman 2 CSP run
python ../bci_framework/run_bcic.py BCICIV2a RANDOM_FOREST logbp 2 CSP run
python ../bci_framework/run_bcic.py BCICIV2a RANDOM_FOREST morlet 2 CSP run
python ../bci_framework/run_bcic.py BCICIV2a RANDOM_FOREST AR 2 CSP run

python ../bci_framework/run_bcic.py BCICIV2a SVM BP 2 CSP run
python ../bci_framework/run_bcic.py BCICIV2a SVM wackerman 2 CSP run
python ../bci_framework/run_bcic.py BCICIV2a SVM logbp 2 CSP run
python ../bci_framework/run_bcic.py BCICIV2a SVM morlet 2 CSP run
python ../bci_framework/run_bcic.py BCICIV2a SVM AR 2 CSP run

python ../bci_framework/run_bcic.py BCICIV2a LogisticRegression BP 2 CSP run
python ../bci_framework/run_bcic.py BCICIV2a LogisticRegression wackerman 2 CSP run
python ../bci_framework/run_bcic.py BCICIV2a LogisticRegression logbp 2 CSP run
python ../bci_framework/run_bcic.py BCICIV2a LogisticRegression morlet 2 CSP run
python ../bci_framework/run_bcic.py BCICIV2a LogisticRegression AR 2 CSP run

python ../bci_framework/run_bcic.py BCICIV2a Boosting BP 2 CSP run
python ../bci_framework/run_bcic.py BCICIV2a Boosting wackerman 2 CSP run
python ../bci_framework/run_bcic.py BCICIV2a Boosting logbp 2 CSP run
python ../bci_framework/run_bcic.py BCICIV2a Boosting morlet 2 CSP run
python ../bci_framework/run_bcic.py BCICIV2a Boosting AR 2 CSP run

python ../bci_framework/run_bcic.py BCICIV2a MLP BP 2 CSP run
python ../bci_framework/run_bcic.py BCICIV2a MLP wackerman 2 CSP run
python ../bci_framework/run_bcic.py BCICIV2a MLP logbp 2 CSP run
python ../bci_framework/run_bcic.py BCICIV2a MLP morlet 2 CSP run
python ../bci_framework/run_bcic.py BCICIV2a MLP AR 2 CSP run

echo 'running BCIC iv2a C34Z channels: '

python ../bci_framework/run_bcic.py BCICIV2a LDA BP -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV2a LDA wackerman -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV2a LDA logbp -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV2a LDA morlet -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV2a LDA AR -1 C34Z run

python ../bci_framework/run_bcic.py BCICIV2a QDA BP -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV2a QDA wackerman -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV2a QDA logbp -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV2a QDA morlet -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV2a QDA AR -1 C34Z run

python ../bci_framework/run_bcic.py BCICIV2a RANDOM_FOREST BP -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV2a RANDOM_FOREST wackerman -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV2a RANDOM_FOREST logbp -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV2a RANDOM_FOREST morlet -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV2a RANDOM_FOREST AR -1 C34Z run

python ../bci_framework/run_bcic.py BCICIV2a SVM BP -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV2a SVM wackerman -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV2a SVM logbp -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV2a SVM morlet -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV2a SVM AR -1 C34Z run

python ../bci_framework/run_bcic.py BCICIV2a LogisticRegression BP -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV2a LogisticRegression wackerman -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV2a LogisticRegression logbp -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV2a LogisticRegression morlet -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV2a LogisticRegression AR -1 C34Z run

python ../bci_framework/run_bcic.py BCICIV2a Boosting BP -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV2a Boosting wackerman -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV2a Boosting logbp -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV2a Boosting morlet -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV2a Boosting AR -1 C34Z run

python ../bci_framework/run_bcic.py BCICIV2a MLP BP -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV2a MLP wackerman -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV2a MLP logbp -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV2a MLP morlet -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV2a MLP AR -1 C34Z run
