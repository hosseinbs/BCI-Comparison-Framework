echo 'evaluating BCIC iv2a: '

python ../bci_framework/run_bcic.py BCICIV2a LDA BP -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2a LDA logbp -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2a LDA morlet -1 ALL evaluate

python ../bci_framework/run_bcic.py BCICIV2a QDA BP -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2a QDA logbp -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2a QDA morlet -1 ALL evaluate

python ../bci_framework/run_bcic.py BCICIV2a RANDOM_FOREST BP -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2a RANDOM_FOREST logbp -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2a RANDOM_FOREST morlet -1 ALL evaluate

python ../bci_framework/run_bcic.py BCICIV2a SVM BP -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2a SVM logbp -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2a SVM morlet -1 ALL evaluate

python ../bci_framework/run_bcic.py BCICIV2a LogisticRegression BP -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2a LogisticRegression logbp -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2a LogisticRegression morlet -1 ALL evaluate

python ../bci_framework/run_bcic.py BCICIV2a Boosting BP -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2a Boosting logbp -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2a Boosting morlet -1 ALL evaluate

python ../bci_framework/run_bcic.py BCICIV2a MLP BP -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2a MLP logbp -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV2a MLP morlet -1 ALL evaluate

