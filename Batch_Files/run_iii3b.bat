echo 'running BCIC iii3b: '

python ../bci_framework/run_bcic.py BCICIII3b LDA BP -1 ALL run
python ../bci_framework/run_bcic.py BCICIII3b LDA wackerman -1 ALL run
python ../bci_framework/run_bcic.py BCICIII3b LDA logbp -1 ALL run
python ../bci_framework/run_bcic.py BCICIII3b LDA morlet -1 ALL run
python ../bci_framework/run_bcic.py BCICIII3b LDA AR -1 ALL run

python ../bci_framework/run_bcic.py BCICIII3b QDA BP -1 ALL run
python ../bci_framework/run_bcic.py BCICIII3b QDA wackerman -1 ALL run
python ../bci_framework/run_bcic.py BCICIII3b QDA logbp -1 ALL run
python ../bci_framework/run_bcic.py BCICIII3b QDA morlet -1 ALL run
python ../bci_framework/run_bcic.py BCICIII3b QDA AR -1 ALL run

python ../bci_framework/run_bcic.py BCICIII3b RANDOM_FOREST BP -1 ALL run
python ../bci_framework/run_bcic.py BCICIII3b RANDOM_FOREST wackerman -1 ALL run
python ../bci_framework/run_bcic.py BCICIII3b RANDOM_FOREST logbp -1 ALL run
python ../bci_framework/run_bcic.py BCICIII3b RANDOM_FOREST morlet -1 ALL run
python ../bci_framework/run_bcic.py BCICIII3b RANDOM_FOREST AR -1 ALL run

python ../bci_framework/run_bcic.py BCICIII3b SVM BP -1 ALL run
python ../bci_framework/run_bcic.py BCICIII3b SVM wackerman -1 ALL run
python ../bci_framework/run_bcic.py BCICIII3b SVM logbp -1 ALL run
python ../bci_framework/run_bcic.py BCICIII3b SVM morlet -1 ALL run
python ../bci_framework/run_bcic.py BCICIII3b SVM AR -1 ALL run

python ../bci_framework/run_bcic.py BCICIII3b LogisticRegression BP -1 ALL run
python ../bci_framework/run_bcic.py BCICIII3b LogisticRegression wackerman -1 ALL run
python ../bci_framework/run_bcic.py BCICIII3b LogisticRegression logbp -1 ALL run
python ../bci_framework/run_bcic.py BCICIII3b LogisticRegression morlet -1 ALL run
python ../bci_framework/run_bcic.py BCICIII3b LogisticRegression AR -1 ALL run

python ../bci_framework/run_bcic.py BCICIII3b Boosting BP -1 ALL run
python ../bci_framework/run_bcic.py BCICIII3b Boosting wackerman -1 ALL run
python ../bci_framework/run_bcic.py BCICIII3b Boosting logbp -1 ALL run
python ../bci_framework/run_bcic.py BCICIII3b Boosting morlet -1 ALL run
python ../bci_framework/run_bcic.py BCICIII3b Boosting AR -1 ALL run

python ../bci_framework/run_bcic.py BCICIII3b MLP BP -1 ALL run
python ../bci_framework/run_bcic.py BCICIII3b MLP wackerman -1 ALL run
python ../bci_framework/run_bcic.py BCICIII3b MLP logbp -1 ALL run
python ../bci_framework/run_bcic.py BCICIII3b MLP morlet -1 ALL run
python ../bci_framework/run_bcic.py BCICIII3b MLP AR -1 ALL run
