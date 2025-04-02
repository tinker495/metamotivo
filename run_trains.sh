pip install humenv/.

#xvfb-run -a python examples/fbcpr_train_humenv.py --compile --motions humenv/data_preparation/test_train_split/large1_small1_train_0.1.txt --motions_root humenv/data_preparation/humenv_amass/ --prioritization
python examples/fbcpr_train_humenv.py --compile --motions humenv/data_preparation/test_train_split/large1_small1_train_0.1.txt --motions_root humenv/data_preparation/humenv_amass/ --prioritization