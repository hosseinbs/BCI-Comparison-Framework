echo 'running SM2: '

python ../bci_framework/run_bcic.py SM2 LDA BP -1 ALL evaluate
python ../bci_framework/run_bcic.py SM2 LDA logbp -1 ALL evaluate
python ../bci_framework/run_bcic.py SM2 LDA morlet -1 ALL evaluate

python ../bci_framework/run_bcic.py SM2 QDA BP -1 ALL evaluate
python ../bci_framework/run_bcic.py SM2 QDA logbp -1 ALL evaluate
python ../bci_framework/run_bcic.py SM2 QDA morlet -1 ALL evaluate

python ../bci_framework/run_bcic.py SM2 RANDOM_FOREST BP -1 ALL evaluate
python ../bci_framework/run_bcic.py SM2 RANDOM_FOREST logbp -1 ALL evaluate
python ../bci_framework/run_bcic.py SM2 RANDOM_FOREST morlet -1 ALL evaluate

python ../bci_framework/run_bcic.py SM2 SVM BP -1 ALL evaluate
python ../bci_framework/run_bcic.py SM2 SVM logbp -1 ALL evaluate
python ../bci_framework/run_bcic.py SM2 SVM morlet -1 ALL evaluate

python ../bci_framework/run_bcic.py SM2 LogisticRegression BP -1 ALL evaluate
python ../bci_framework/run_bcic.py SM2 LogisticRegression logbp -1 ALL evaluate
python ../bci_framework/run_bcic.py SM2 LogisticRegression morlet -1 ALL evaluate

python ../bci_framework/run_bcic.py SM2 Boosting BP -1 ALL evaluate
python ../bci_framework/run_bcic.py SM2 Boosting logbp -1 ALL evaluate
python ../bci_framework/run_bcic.py SM2 Boosting morlet -1 ALL evaluate

python ../bci_framework/run_bcic.py SM2 MLP BP -1 ALL evaluate
python ../bci_framework/run_bcic.py SM2 MLP logbp -1 ALL evaluate
python ../bci_framework/run_bcic.py SM2 MLP morlet -1 ALL evaluate

