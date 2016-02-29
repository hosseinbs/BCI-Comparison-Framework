echo 'evaluatening BCIC iv1: '

python ../bci_framework/run_bcic.py BCICIV1 LDA BP -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV1 LDA logbp -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV1 LDA morlet -1 ALL evaluate

python ../bci_framework/run_bcic.py BCICIV1 QDA BP -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV1 QDA logbp -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV1 QDA morlet -1 ALL evaluate

python ../bci_framework/run_bcic.py BCICIV1 RANDOM_FOREST BP -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV1 RANDOM_FOREST logbp -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV1 RANDOM_FOREST morlet -1 ALL evaluate

python ../bci_framework/run_bcic.py BCICIV1 SVM BP -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV1 SVM logbp -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV1 SVM morlet -1 ALL evaluate

python ../bci_framework/run_bcic.py BCICIV1 LogisticRegression BP -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV1 LogisticRegression logbp -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV1 LogisticRegression morlet -1 ALL evaluate

python ../bci_framework/run_bcic.py BCICIV1 Boosting BP -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV1 Boosting logbp -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV1 Boosting morlet -1 ALL evaluate

python ../bci_framework/run_bcic.py BCICIV1 MLP BP -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV1 MLP logbp -1 ALL evaluate
python ../bci_framework/run_bcic.py BCICIV1 MLP morlet -1 ALL evaluate
