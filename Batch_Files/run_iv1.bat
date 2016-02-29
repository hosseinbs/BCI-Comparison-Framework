echo 'running BCIC iv1: '

python ../bci_framework/run_bcic.py BCICIV1 LDA BP -1 ALL run
python ../bci_framework/run_bcic.py BCICIV1 LDA logbp -1 ALL run
python ../bci_framework/run_bcic.py BCICIV1 LDA morlet -1 ALL run
python ../bci_framework/run_bcic.py BCICIV1 LDA AR -1 ALL run

python ../bci_framework/run_bcic.py BCICIV1 QDA BP -1 ALL run
python ../bci_framework/run_bcic.py BCICIV1 QDA logbp -1 ALL run
python ../bci_framework/run_bcic.py BCICIV1 QDA morlet -1 ALL run
python ../bci_framework/run_bcic.py BCICIV1 QDA AR -1 ALL run

python ../bci_framework/run_bcic.py BCICIV1 RANDOM_FOREST BP -1 ALL run
python ../bci_framework/run_bcic.py BCICIV1 RANDOM_FOREST logbp -1 ALL run
python ../bci_framework/run_bcic.py BCICIV1 RANDOM_FOREST morlet -1 ALL run
python ../bci_framework/run_bcic.py BCICIV1 RANDOM_FOREST AR -1 ALL run

python ../bci_framework/run_bcic.py BCICIV1 SVM BP -1 ALL run
python ../bci_framework/run_bcic.py BCICIV1 SVM logbp -1 ALL run
python ../bci_framework/run_bcic.py BCICIV1 SVM morlet -1 ALL run
python ../bci_framework/run_bcic.py BCICIV1 SVM AR -1 ALL run

python ../bci_framework/run_bcic.py BCICIV1 LogisticRegression BP -1 ALL run
python ../bci_framework/run_bcic.py BCICIV1 LogisticRegression logbp -1 ALL run
python ../bci_framework/run_bcic.py BCICIV1 LogisticRegression morlet -1 ALL run
python ../bci_framework/run_bcic.py BCICIV1 LogisticRegression AR -1 ALL run

python ../bci_framework/run_bcic.py BCICIV1 Boosting BP -1 ALL run
python ../bci_framework/run_bcic.py BCICIV1 Boosting logbp -1 ALL run
python ../bci_framework/run_bcic.py BCICIV1 Boosting morlet -1 ALL run
python ../bci_framework/run_bcic.py BCICIV1 Boosting AR -1 ALL run

python ../bci_framework/run_bcic.py BCICIV1 MLP BP -1 ALL run
python ../bci_framework/run_bcic.py BCICIV1 MLP logbp -1 ALL run
python ../bci_framework/run_bcic.py BCICIV1 MLP morlet -1 ALL run
python ../bci_framework/run_bcic.py BCICIV1 MLP AR -1 ALL run

echo 'running BCIC iv1 with CSP: '

python ../bci_framework/run_bcic.py BCICIV1 LDA BP 4 CSP run
python ../bci_framework/run_bcic.py BCICIV1 LDA logbp 4 CSP run
python ../bci_framework/run_bcic.py BCICIV1 LDA morlet 4 CSP run
python ../bci_framework/run_bcic.py BCICIV1 LDA AR 4 CSP run

python ../bci_framework/run_bcic.py BCICIV1 QDA BP 4 CSP run
python ../bci_framework/run_bcic.py BCICIV1 QDA logbp 4 CSP run
python ../bci_framework/run_bcic.py BCICIV1 QDA morlet 4 CSP run
python ../bci_framework/run_bcic.py BCICIV1 QDA AR 4 CSP run

python ../bci_framework/run_bcic.py BCICIV1 RANDOM_FOREST BP 4 CSP run
python ../bci_framework/run_bcic.py BCICIV1 RANDOM_FOREST logbp 4 CSP run
python ../bci_framework/run_bcic.py BCICIV1 RANDOM_FOREST morlet 4 CSP run
python ../bci_framework/run_bcic.py BCICIV1 RANDOM_FOREST AR 4 CSP run

python ../bci_framework/run_bcic.py BCICIV1 SVM BP 4 CSP run
python ../bci_framework/run_bcic.py BCICIV1 SVM logbp 4 CSP run
python ../bci_framework/run_bcic.py BCICIV1 SVM morlet 4 CSP run
python ../bci_framework/run_bcic.py BCICIV1 SVM AR 4 CSP run

python ../bci_framework/run_bcic.py BCICIV1 LogisticRegression BP 4 CSP run
python ../bci_framework/run_bcic.py BCICIV1 LogisticRegression logbp 4 CSP run
python ../bci_framework/run_bcic.py BCICIV1 LogisticRegression morlet 4 CSP run
python ../bci_framework/run_bcic.py BCICIV1 LogisticRegression AR 4 CSP run

python ../bci_framework/run_bcic.py BCICIV1 Boosting BP 4 CSP run
python ../bci_framework/run_bcic.py BCICIV1 Boosting logbp 4 CSP run
python ../bci_framework/run_bcic.py BCICIV1 Boosting morlet 4 CSP run
python ../bci_framework/run_bcic.py BCICIV1 Boosting AR 4 CSP run

python ../bci_framework/run_bcic.py BCICIV1 MLP BP 4 CSP run
python ../bci_framework/run_bcic.py BCICIV1 MLP logbp 4 CSP run
python ../bci_framework/run_bcic.py BCICIV1 MLP morlet 4 CSP run
python ../bci_framework/run_bcic.py BCICIV1 MLP AR 4 CSP run

echo 'running BCIC iv1 C34Z channels: '

python ../bci_framework/run_bcic.py BCICIV1 LDA BP -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV1 LDA logbp -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV1 LDA morlet -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV1 LDA AR -1 C34Z run

python ../bci_framework/run_bcic.py BCICIV1 QDA BP -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV1 QDA logbp -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV1 QDA morlet -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV1 QDA AR -1 C34Z run

python ../bci_framework/run_bcic.py BCICIV1 RANDOM_FOREST BP -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV1 RANDOM_FOREST logbp -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV1 RANDOM_FOREST morlet -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV1 RANDOM_FOREST AR -1 C34Z run

python ../bci_framework/run_bcic.py BCICIV1 SVM BP -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV1 SVM logbp -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV1 SVM morlet -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV1 SVM AR -1 C34Z run

python ../bci_framework/run_bcic.py BCICIV1 LogisticRegression BP -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV1 LogisticRegression logbp -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV1 LogisticRegression morlet -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV1 LogisticRegression AR -1 C34Z run

python ../bci_framework/run_bcic.py BCICIV1 Boosting BP -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV1 Boosting logbp -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV1 Boosting morlet -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV1 Boosting AR -1 C34Z run

python ../bci_framework/run_bcic.py BCICIV1 MLP BP -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV1 MLP logbp -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV1 MLP morlet -1 C34Z run
python ../bci_framework/run_bcic.py BCICIV1 MLP AR -1 C34Z run
